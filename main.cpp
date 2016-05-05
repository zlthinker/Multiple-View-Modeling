#include <iostream>
#include "GCO/energy.h"
#include "GCO/GCoptimization.h"
#include <opencv/highgui.h>

using namespace std;

void NormalOptimization(std::vector<cv::Vec3d> & estim_norm, std::vector<cv::Vec3d> & label_norm, std::vector<int> & label_id,
                        int width, int height, std::vector<int> & results);
int GCO_main(int argc, char **argv);

int main() {

    /*
    int width = 10;
    int height = 5;
    int num_pixels = width*height;
    int num_labels = 7;

    std::vector<cv::Vec3d> estim_norms;
    std::vector<cv::Vec3d> label_norms;
    std::vector<int> label_ids;
    std::vector<int> results;
    estim_norms.reserve(num_pixels);
    label_norms.reserve(num_labels);
    label_ids.reserve(num_pixels);
    results.reserve(num_pixels);

    for (int i = 0; i < num_pixels; i++)
    {
        estim_norms.push_back(cv::Vec3d(i, i, i));
        label_ids.push_back(i % num_labels);
    }

    for (int i = 0; i < num_labels; i++)
    {
        label_norms.push_back(cv::Vec3d(i, i, i));
    }

    NormalOptimization(estim_norms, label_norms, label_ids, width, height, results);
     */

    char ** c;
    GCO_main(0, c);




    return 0;
}