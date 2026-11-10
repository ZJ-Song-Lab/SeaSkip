// SeaSkip board orchestrator (Atlas 200I DK, Ascend310B4, CANN 7.0.RC1)
// Paper experiments M3 (gate skip quality) + M6 (edge latency / compute saving).
//
// v4 pipeline per patch (640x640 JPG):
//   decode JPG (stb_image) [decode_ms] -> decimation [0::4,0::4] (1 read/px)
//   -> 160x160 NCHW gate
//   input [gprep_ms] -> gate.om (3-way logits) [gate_ms] -> softmax -> decision:
//        mode=full : always build 640 NCHW [prep_ms] + run det.om [det_ms]
//        mode=skip : run det only if gate != empty; empty skips prep+det entirely
//        mode=gate : never run det.om (pure gate latency / decisions)
//   det.om output [1,5,8400] = cx,cy,w,h (pixels) + ship score (post-sigmoid)
//   -> conf 0.25 -> greedy NMS IoU 0.45 -> boxes in patch pixel coords
//
// CSV out: idx,file,gate_pred,p_empty,p_simple,p_complex,ran_det,n_det,boxes,
//          prep_ms,gate_ms,det_ms,decode_ms,gprep_ms
// (v3: prep_ms = full-size 640 NCHW build, 0 when skipped; decode/gprep always
//  paid; boxes format "x1,y1,x2,y2,conf|..." in patch pixel coords)
//
// Build (on board):
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   g++ -O2 -std=c++17 seaskip_eval.cpp -o seaskip_eval \
//       -I$ASCEND_TOOLKIT_HOME/include -L$ASCEND_TOOLKIT_HOME/lib64 -lascendcl
// Run:
//   ./seaskip_eval gate.om det.om patches_dir result.csv <full|skip|gate>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "acl/acl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#define CHECK_OK(expr, msg)                                                    \
    do {                                                                       \
        if ((expr) != ACL_SUCCESS) {                                            \
            fprintf(stderr, "[FATAL] %s (aclret=%d) at %s:%d\n", msg,          \
                    (int)(expr), __FILE__, __LINE__);                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static const int kImg = 640;
static const int kGate = 160;  // v4 gate input (stride-4 decimation of kImg)
static const float kConfThres = 0.25f;
static const float kNmsThres = 0.45f;

struct Model {
    uint32_t id = 0;
    aclmdlDesc *desc = nullptr;
    aclmdlDataset *in = nullptr;
    aclmdlDataset *out = nullptr;
    void *in_dev = nullptr;
    void *out_dev = nullptr;
    void *in_host = nullptr;
    void *out_host = nullptr;
    size_t in_size = 0;
    size_t out_size = 0;
};

static void load_model(Model &m, const char *path) {
    CHECK_OK(aclmdlLoadFromFile(path, &m.id), "aclmdlLoadFromFile");
    m.desc = aclmdlCreateDesc();
    CHECK_OK(aclmdlGetDesc(m.desc, m.id), "aclmdlGetDesc");
    m.in_size = aclmdlGetInputSizeByIndex(m.desc, 0);
    m.out_size = aclmdlGetOutputSizeByIndex(m.desc, 0);
    CHECK_OK(aclrtMalloc(&m.in_dev, m.in_size, ACL_MEM_MALLOC_NORMAL_ONLY),
             "aclrtMalloc in_dev");
    CHECK_OK(aclrtMalloc(&m.out_dev, m.out_size, ACL_MEM_MALLOC_NORMAL_ONLY),
             "aclrtMalloc out_dev");
    CHECK_OK(aclrtMallocHost(&m.in_host, m.in_size), "aclrtMallocHost in");
    CHECK_OK(aclrtMallocHost(&m.out_host, m.out_size), "aclrtMallocHost out");
    m.in = aclmdlCreateDataset();
    aclDataBuffer *ib = aclCreateDataBuffer(m.in_dev, m.in_size);
    CHECK_OK(aclmdlAddDatasetBuffer(m.in, ib), "add input buffer");
    m.out = aclmdlCreateDataset();
    aclDataBuffer *ob = aclCreateDataBuffer(m.out_dev, m.out_size);
    CHECK_OK(aclmdlAddDatasetBuffer(m.out, ob), "add output buffer");
    printf("[model] %s in=%zuB out=%zuB\n", path, m.in_size, m.out_size);
}

static void destroy_model(Model &m) {
    if (m.in) {
        aclDataBuffer *ib = aclmdlGetDatasetBuffer(m.in, 0);
        if (ib) aclDestroyDataBuffer(ib);
        aclmdlDestroyDataset(m.in);
    }
    if (m.out) {
        aclDataBuffer *ob = aclmdlGetDatasetBuffer(m.out, 0);
        if (ob) aclDestroyDataBuffer(ob);
        aclmdlDestroyDataset(m.out);
    }
    if (m.in_dev) aclrtFree(m.in_dev);
    if (m.out_dev) aclrtFree(m.out_dev);
    if (m.in_host) aclrtFreeHost(m.in_host);
    if (m.out_host) aclrtFreeHost(m.out_host);
    if (m.desc) aclmdlDestroyDesc(m.desc);
    aclmdlUnload(m.id);
}

// Run one model. Returns execution time in ms (device sync included).
static float run_model(Model &m) {
    CHECK_OK(aclrtMemcpy(m.in_dev, m.in_size, m.in_host, m.in_size,
                         ACL_MEMCPY_HOST_TO_DEVICE), "memcpy H2D");
    auto t0 = Clock::now();
    CHECK_OK(aclmdlExecute(m.id, m.in, m.out), "aclmdlExecute");
    CHECK_OK(aclrtMemcpy(m.out_host, m.out_size, m.out_dev, m.out_size,
                         ACL_MEMCPY_DEVICE_TO_HOST), "memcpy D2H");
    auto t1 = Clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

struct DetBox {
    float x1, y1, x2, y2, score;
};

static float iou(const DetBox &a, const DetBox &b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1)
                - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s gate.om det.om patches_dir out.csv <full|skip|gate>\n",
                argv[0]);
        return 1;
    }
    const char *gate_path = argv[1];
    const char *det_path = argv[2];
    const fs::path dir = argv[3];
    const char *out_csv = argv[4];
    std::string mode = argv[5];
    if (mode != "full" && mode != "skip" && mode != "gate") {
        fprintf(stderr, "[FATAL] mode must be full|skip|gate\n");
        return 1;
    }

    // Collect patch list in deterministic (lexicographic) order.
    std::vector<std::string> files;
    for (const auto &e : fs::directory_iterator(dir)) {
        std::string name = e.path().filename().string();
        std::string low = name;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (low.size() > 4 && low.substr(low.size() - 4) == ".jpg")
            files.push_back(name);
    }
    if (files.empty()) {
        fprintf(stderr, "[FATAL] no .jpg under %s\n", dir.string().c_str());
        return 1;
    }
    std::sort(files.begin(), files.end());
    printf("[stream] %zu patches, mode=%s\n", files.size(), mode.c_str());

    CHECK_OK(aclInit(nullptr), "aclInit");
    CHECK_OK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtContext ctx;
    CHECK_OK(aclrtCreateContext(&ctx, 0), "aclrtCreateContext");
    aclrtStream stream;
    CHECK_OK(aclrtCreateStream(&stream), "aclrtCreateStream");
    (void)stream;

    Model gate, det;
    load_model(gate, gate_path);
    if (mode != "gate") load_model(det, det_path);

    // Sanity: gate input must be 1x3x160x160 float32 (307,200 B) for v3.
    if (gate.in_size != (size_t)3 * kGate * kGate * 4) {
        fprintf(stderr, "[FATAL] gate input size %zu != 307200 "
                        "(need 160x160 v3 gate om)\n", gate.in_size);
        return 1;
    }

    // Warmup each loaded model 3x on zeros (excluded from timing).
    memset(gate.in_host, 0, gate.in_size);
    for (int i = 0; i < 3; ++i) run_model(gate);
    if (mode != "gate") {
        if (det.in_size != (size_t)3 * kImg * kImg * 4) {
            fprintf(stderr, "[FATAL] det input size %zu != 3072000\n",
                    det.in_size);
            return 1;
        }
        memset(det.in_host, 0, det.in_size);
        for (int i = 0; i < 3; ++i) run_model(det);
    }

    FILE *csv = fopen(out_csv, "w");
    if (!csv) {
        fprintf(stderr, "[FATAL] cannot open %s\n", out_csv);
        return 1;
    }
    fprintf(csv,
            "idx,file,gate_pred,p_empty,p_simple,p_complex,ran_det,n_det,"
            "boxes,prep_ms,gate_ms,det_ms,decode_ms,gprep_ms\n");

    double sum_prep = 0, sum_gate = 0, sum_det = 0, sum_dec = 0, sum_gp = 0;
    long n_det_runs = 0, n_skipped = 0;
    unsigned char *pixels = nullptr;

    for (size_t idx = 0; idx < files.size(); ++idx) {
        fs::path p = dir / files[idx];
        auto tdec0 = Clock::now();
        int w = 0, h = 0, c = 0;
        pixels = stbi_load(p.string().c_str(), &w, &h, &c, 3);
        if (!pixels) {
            fprintf(stderr, "[WARN] decode failed: %s\n", p.string().c_str());
            continue;
        }
        if (w != kImg || h != kImg) {
            fprintf(stderr, "[WARN] size %dx%d != 640: %s\n", w, h,
                    p.string().c_str());
            stbi_image_free(pixels);
            continue;
        }
        auto tdec1 = Clock::now();
        float decode_ms =
            std::chrono::duration<float, std::milli>(tdec1 - tdec0).count();

        // v4: build 160x160 gate input = pure decimation (1 read per output
        // pixel), bitwise-aligned with training-side numpy [0::4,0::4]
        auto tgp0 = Clock::now();
        float *gbuf = (float *)gate.in_host;
        const size_t gplane = (size_t)kGate * kGate;
        for (int gy = 0; gy < kGate; ++gy)
            for (int gx = 0; gx < kGate; ++gx) {
                const unsigned char *p = pixels
                    + 3 * ((size_t)(gy * 4) * kImg + (size_t)(gx * 4));
                float *d = gbuf + (size_t)gy * kGate + gx;
                d[0] = (float)p[0] / 255.0f;
                d[gplane] = (float)p[1] / 255.0f;
                d[2 * gplane] = (float)p[2] / 255.0f;
            }
        auto tgp1 = Clock::now();
        float gprep_ms =
            std::chrono::duration<float, std::milli>(tgp1 - tgp0).count();

        // ---- gate inference ----
        float gate_ms = run_model(gate);
        float *logits = (float *)gate.out_host;  // 3 logits: empty/simple/complex
        float mx = std::max(logits[0], std::max(logits[1], logits[2]));
        float e0 = expf(logits[0] - mx), e1 = expf(logits[1] - mx),
              e2 = expf(logits[2] - mx);
        float z = e0 + e1 + e2;
        float p0 = e0 / z, p1 = e1 / z, p2 = e2 / z;
        int pred = (p0 >= p1 && p0 >= p2) ? 0 : ((p1 >= p2) ? 1 : 2);

        int ran_det = 0, n_det = 0;
        char boxes_buf[1 << 16] = "";
        float det_ms = 0.0f, prep_ms = 0.0f;

        bool need_det = (mode == "full") || (mode == "skip" && pred != 0);
        if (need_det && mode != "gate") {
            // v3: full-size 640 NCHW is built ONLY when det will run
            // (empty patches skip this ~15ms CPU loop entirely)
            auto tp0 = Clock::now();
            float *fbuf = (float *)det.in_host;
            const size_t plane = (size_t)kImg * kImg;
            for (int ch = 0; ch < 3; ++ch)
                for (size_t i = 0; i < plane; ++i)
                    fbuf[ch * plane + i] = pixels[3 * i + ch] / 255.0f;
            auto tp1 = Clock::now();
            prep_ms =
                std::chrono::duration<float, std::milli>(tp1 - tp0).count();
            det_ms = run_model(det);
            ran_det = 1;
            ++n_det_runs;
            // decode [1,5,8400]
            const float *o = (const float *)det.out_host;
            const int n_anc = (int)(det.out_size / 4 / 5);
            std::vector<DetBox> cand;
            for (int i = 0; i < n_anc; ++i) {
                float conf = o[4 * (size_t)n_anc + i];
                if (conf < kConfThres) continue;
                float cx = o[0 * (size_t)n_anc + i], cy = o[1 * (size_t)n_anc + i];
                float bw = o[2 * (size_t)n_anc + i], bh = o[3 * (size_t)n_anc + i];
                DetBox b{(cx - bw / 2), (cy - bh / 2), (cx + bw / 2),
                         (cy + bh / 2), conf};
                if (b.x1 < -8 || b.y1 < -8 || b.x2 > kImg + 8 || b.y2 > kImg + 8
                    || b.x2 <= b.x1 || b.y2 <= b.y1)
                    continue;
                cand.push_back(b);
            }
            std::sort(cand.begin(), cand.end(),
                      [](const DetBox &a, const DetBox &b) {
                          return a.score > b.score;
                      });
            std::vector<char> keep(cand.size(), 1);
            std::vector<DetBox> kept;
            for (size_t i = 0; i < cand.size(); ++i) {
                if (!keep[i]) continue;
                kept.push_back(cand[i]);
                for (size_t j = i + 1; j < cand.size(); ++j)
                    if (keep[j] && iou(cand[i], cand[j]) > kNmsThres)
                        keep[j] = 0;
            }
            n_det = (int)kept.size();
            size_t off = 0;
            for (size_t i = 0; i < kept.size(); ++i) {
                int nwr = snprintf(boxes_buf + off, sizeof(boxes_buf) - off,
                                   "%s%.1f,%.1f,%.1f,%.1f,%.3f",
                                   i ? "|" : "", kept[i].x1, kept[i].y1,
                                   kept[i].x2, kept[i].y2, kept[i].score);
                if (nwr <= 0 || off + (size_t)nwr >= sizeof(boxes_buf) - 1)
                    break;
                off += (size_t)nwr;
            }
        } else if (mode == "skip" && pred == 0) {
            ++n_skipped;  // empty: full-size prep + det entirely skipped
        }
        stbi_image_free(pixels);

        sum_dec += decode_ms;
        sum_gp += gprep_ms;
        sum_prep += prep_ms;
        sum_gate += gate_ms;
        sum_det += det_ms;
        fprintf(csv,
                "%zu,%s,%d,%.4f,%.4f,%.4f,%d,%d,\"%s\",%.3f,%.3f,%.3f,"
                "%.3f,%.3f\n",
                idx, files[idx].c_str(), pred, p0, p1, p2, ran_det, n_det,
                boxes_buf, prep_ms, gate_ms, det_ms, decode_ms, gprep_ms);
        if ((idx + 1) % 50 == 0)
            printf("[progress] %zu/%zu\n", idx + 1, files.size());
    }

    fclose(csv);
    size_t n = files.size();
    printf("[done] n=%zu skipped=%ld det_runs=%ld\n", n, n_skipped, n_det_runs);
    double dec = n ? sum_dec / n : 0, gp = n ? sum_gp / n : 0,
           gt = n ? sum_gate / n : 0;
    double pr = n_det_runs ? sum_prep / n_det_runs : 0;  // per det-run patch
    double dt = n_det_runs ? sum_det / n_det_runs : 0;
    printf("[avg] decode=%.2fms gprep=%.2fms gate=%.2fms prep(ran)=%.2fms "
           "det=%.2fms\n", dec, gp, gt, pr, dt);
    double baseline = dec + pr + dt;  // no-gate system per patch
    double run_rate = n ? (double)n_det_runs / (double)n : 1.0;
    double v3_full = dec + gp + gt + pr + dt;
    double v3_skip = dec + gp + gt + run_rate * (pr + dt);
    printf("[pipe] baseline(no-gate)=%.2fms v3_full=%.2fms v3_skip=%.2fms "
           "(run_rate=%.3f) save_vs_base=%.1f%%\n",
           baseline, v3_full, v3_skip, run_rate,
           baseline > 0 ? 100.0 * (1.0 - v3_skip / baseline) : 0.0);

    if (mode != "gate") destroy_model(det);
    destroy_model(gate);
    CHECK_OK(aclrtDestroyStream(stream), "destroy stream");
    CHECK_OK(aclrtDestroyContext(ctx), "destroy context");
    CHECK_OK(aclrtResetDevice(0), "reset device");
    CHECK_OK(aclFinalize(), "aclFinalize");
    return 0;
}