#ifndef DETECTION_H
#define DETECTION_H

typedef enum {
    ONNC_RUNTIME_DETECTION_NONE = 0,
    ONNC_RUNTIME_DETECTION_YOLO,
} Detection_mode;

typedef struct{
    float x, y, w, h;
} Box;

typedef struct{
    Box bbox;
    int classes;
    float *prob;
    float objectness;
    int sort_class;
} Detection;

struct yolo_result {
    int left, right, top, bottom;
    int category;
};

int get_detection_mode(const char *type);

struct yolo_result *detection(Detection_mode mode, float *values, int class_detection);

#endif