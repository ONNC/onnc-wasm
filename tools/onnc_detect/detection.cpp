#include <detection.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>

#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"

using namespace cimg_library;

static struct yolo_result *yolo_detection(float *values, int detect_class);

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;

    return right - left;
}

static float box_intersection(Box a, Box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w * h;

    return area;
}

static float box_union(Box a, Box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;

    return u;
}

static float box_iou(Box a, Box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

static int nms_comparator(const Detection *a, const Detection *b)
{
    float diff = 0;
    if(b->sort_class >= 0) {
        diff = a->prob[b->sort_class] - b->prob[b->sort_class];
    } else {
        diff = a->objectness - b->objectness;
    }
    return (diff < 0) - (diff > 0);
}

int get_detection_mode(const char *mode) {
    if(!strcmp(mode, "yolo")) return ONNC_RUNTIME_DETECTION_YOLO;
    return ONNC_RUNTIME_DETECTION_NONE;
}

#define CLASS_ALL (20)
struct yolo_result *detection(Detection_mode mode, float *values, int detect_class) {
    if(mode == ONNC_RUNTIME_DETECTION_YOLO) {
        return yolo_detection(values, detect_class);
    }
    return NULL;
}

#define HEIGHT (448)
#define WIDTH  (448)
#define RESULT_BYTES (512)

const char *names[] = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
};

struct yolo_result *yolo_detection(float values[], int detect_class) {
    // TODO: Extract config to file
    struct yolo_result *result = (struct yolo_result *)malloc(sizeof(struct yolo_result));
    const int layer_n = 2;
    const int classes_all = 20;
    const int side = 7;
    const int layer_width = 7;
    const int layer_height = 7;
    const int sqrt = 2;
    const float thresh = 0.1;
    const int num_detections = layer_width * layer_height * layer_n;

    // Setting classes
    int classes = classes_all;
    // Make detections
    Detection dets[num_detections];
    memset(dets, 0, num_detections * sizeof(Detection));
    for(int i = 0; i < num_detections; ++i) {
        dets[i].prob = (float*)calloc(classes, sizeof(float));
    }

    // Get detections
    for (int i = 0; i < side * side; ++i) {
        for(int n = 0; n < layer_n; ++n) {
            int index = i * layer_n + n;
            int box_index = side * side * (classes + layer_n) + index * 4;
            dets[index].bbox.x = (values[box_index + 0] + i % side) / side;
            dets[index].bbox.y = (values[box_index + 1] + i / side) / side;
            dets[index].bbox.w = pow(values[box_index + 2], sqrt);
            dets[index].bbox.h = pow(values[box_index + 3], sqrt);
            dets[index].objectness = values[side * side * classes + index];
            for(int j = 0; j < classes; ++j) {
                float prob = dets[index].objectness * values[i * classes + j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }

    // do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);
    int total = side * side * layer_n;
    for(int i = 0, k = total - 1; i <= k; ++i) {
        if(dets[i].objectness == 0) {
            Detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }

    for(int k = 0; k < classes; ++k) {
        for(int i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }

        qsort(dets,
              total,
              sizeof(Detection),
              (int(*)(const void*, const void*)) nms_comparator);

        for(int i = 0; i < total; ++i) {
            if(dets[i].prob[k] == 0) {
                continue;
            }

            for(int j = i + 1; j < total; ++j) {
                if (box_iou(dets[i].bbox, dets[j].bbox) <= thresh) {
                    continue;
                }
                dets[j].prob[k] = 0;
            }
        }
    }

    // Output
    for(int i = 0, item_class = -1; i < total; ++i, item_class = -1) {
        for(int j = 0; j < classes; ++j) {
            if (!(dets[i].prob[j] > thresh && item_class < 0)) {
                continue;
            }

            printf("%s: %.0f%%, ", names[j], dets[i].prob[j] * 100);
            item_class = j;
        }

        if(item_class < 0) {
            continue;
        }
        printf("(%.3f, %.3f)[%.3f, %.3f]\n", dets[i].bbox.x,
                                             dets[i].bbox.y,
                                             dets[i].bbox.w,
                                             dets[i].bbox.h);

        // Refer from
        // https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/image.c
        int left  = (dets[i].bbox.x - dets[i].bbox.w/2.) * WIDTH;
        int right = (dets[i].bbox.x + dets[i].bbox.w/2.) * WIDTH;
        int top   = (dets[i].bbox.y - dets[i].bbox.h/2.) * HEIGHT;
        int bot   = (dets[i].bbox.y + dets[i].bbox.h/2.) * HEIGHT;

        if(left < 0) {
            left = 0;
        }

        if(right > WIDTH -1) {
            right = WIDTH - 1;
        }

        if(top < 0) {
            top = 0;
        }

        if(bot > HEIGHT - 1) {
            bot = HEIGHT - 1;
        }
        result->category = item_class;
        result->left = left;
        result->top = top;
        result->right = right;
        result->bottom = bot;
        return result;  // TODO: support more than one
    }
    result->category = -1;
    result->left = 0;
    result->top = 0;
    result->right = 0;
    result->bottom = 0;
    return result;
}

int main(int argc, char const *argv[])
{
    if(argc != 4){
        printf("Usage: ./detection <result_file> <image_file> <output_file>\n");
        return -1;
    }
    // Result file
    FILE *fin = fopen(argv[1], "rb");
    float data[1470];
    fread(data, sizeof(float), 1470, fin);
    fclose(fin);

    // Image file
    CImg<uint8_t> in_image(argv[2]);
    CImg<uint8_t> out_image = in_image;

    // Result
    struct yolo_result* result = yolo_detection(data, 20);
    uint8_t bg_color[3] = {255, 255, 255};
    uint8_t fg_color[3] = {0, 0, 0};
    if(result->category >= 0){
        out_image.draw_text(result->left, result->top - 18, names[result->category], fg_color, bg_color, 1, 18);
        out_image.draw_rectangle(result->left, result->top, result->right, result->bottom, bg_color, 1.0f, ~0U);
    }
    out_image.save(argv[3]);
    // Clean
    free(result);
    return 0;
}