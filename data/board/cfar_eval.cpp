// C12 board-side CA-CFAR evaluator: per-patch detection count + timing.
// Protocol matches PC eval_cfar_baseline.py: guard=4, train_half=12 (544 train cells),
// det = pixel > alpha * local_clutter_mean; run iff n_det >= min_pix.
// usage: ./cfar_eval patches_dir out.csv alpha min_pix
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include "stb_image.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s patches_dir out.csv alpha min_pix\n", argv[0]); return 1; }
    const fs::path dir = argv[1];
    const char *out_csv = argv[2];
    const double alpha = atof(argv[3]);
    const int min_pix = atoi(argv[4]);

    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(dir))
        if (e.path().extension() == ".jpg") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    fprintf(stderr, "[stream] %zu patches\n", files.size());

    FILE *f = fopen(out_csv, "w");
    fprintf(f, "idx,file,n_det,run,cfar_ms\n");
    const int G = 4, T = 16;   // guard half-width, outer half-width (G+TRAIN_HALF)
    const int NTRAIN = (2 * T + 1) * (2 * T + 1) - (2 * G + 1) * (2 * G + 1); // 544

    std::vector<uint32_t> ii;           // integral image (W+1)*(H+1)
    double t_sum = 0;
    for (size_t idx = 0; idx < files.size(); idx++) {
        int w, h, c;
        unsigned char *img = stbi_load(files[idx].string().c_str(), &w, &h, &c, 1);
        if (!img) { fprintf(stderr, "[warn] decode fail %s\n", files[idx].string().c_str()); continue; }
        auto t0 = std::chrono::steady_clock::now();
        // integral image
        ii.assign((size_t)(w + 1) * (h + 1), 0);
        for (int y = 0; y < h; y++) {
            uint32_t rowsum = 0;
            for (int x = 0; x < w; x++) {
                rowsum += img[(size_t)y * w + x];
                ii[(size_t)(y + 1) * (w + 1) + (x + 1)] = ii[(size_t)y * (w + 1) + (x + 1)] + rowsum;
            }
        }
        auto box = [&](int x0, int y0, int x1, int y1) -> uint32_t { // [x0,x1) x [y0,y1)
            return ii[(size_t)y1 * (w + 1) + x1] - ii[(size_t)y0 * (w + 1) + x1]
                 - ii[(size_t)y1 * (w + 1) + x0] + ii[(size_t)y0 * (w + 1) + x0];
        };
        int n_det = 0;
        for (int y = T; y + T < h; y++) {
            for (int x = T; x + T < w; x++) {
                uint32_t outer = box(x - T, y - T, x + T + 1, y + T + 1);
                uint32_t inner = box(x - G, y - G, x + G + 1, y + G + 1);
                double clutter = (double)(outer - inner) / NTRAIN;
                if (img[(size_t)y * w + x] > alpha * clutter) n_det++;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_sum += ms;
        fprintf(f, "%zu,%s,%d,%d,%.3f\n", idx, files[idx].filename().string().c_str(),
                n_det, n_det >= min_pix ? 1 : 0, ms);
        stbi_image_free(img);
        if ((idx + 1) % 1000 == 0) fprintf(stderr, "[progress] %zu/%zu avg=%.2fms\n", idx + 1, files.size(), t_sum / (idx + 1));
    }
    fclose(f);
    fprintf(stderr, "[done] avg cfar=%.2fms\n", t_sum / files.size());
    return 0;
}