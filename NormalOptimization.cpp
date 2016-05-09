//
// Created by Larry CHOU on 5/4/16.
//

#include "NormalOptimization.h"
#include "GCO/GCoptimization.h"

struct NormDataFn{
    std::vector<cv::Vec3d> & estim_norm;
    std::vector<cv::Vec3d> & label_norm;
    NormDataFn(std::vector<cv::Vec3d> & estim_norm, std::vector<cv::Vec3d> & label_norm) :
            estim_norm(estim_norm), label_norm(label_norm) {}
};

double DataFn(int p, int l, void* data)
{
    NormDataFn* norm_data = (NormDataFn*) data;
    cv::Vec3d estim_norm = (norm_data->estim_norm)[p];
    cv::Vec3d label_norm = (norm_data->label_norm)[l];
    return cv::norm(estim_norm - label_norm);
            //std::sqrt(pow(estim_norm[0] - label_norm[0], 2) + pow(estim_norm[1] - label_norm[1], 2) + pow(estim_norm[1] - label_norm[1], 2));
}

double SmoothFn(int p1, int p2, int l1, int l2, void* data)
{
    double lambda = 0.01;
    double sigma = 1;
    NormDataFn* norm_data = (NormDataFn*) data;
    cv::Vec3d label_norm1 = (norm_data->label_norm)[l1];
    cv::Vec3d label_norm2 = (norm_data->label_norm)[l2];
    return lambda * log(1 + cv::norm(label_norm1 - label_norm2) / (2 * pow(sigma, 2)));
}

void NormalOptimization(std::vector<cv::Vec3d> & estim_norm, std::vector<cv::Vec3d> & label_norm,
                        int width, int height, std::vector<int> & results)
{
    int num_pixels = estim_norm.size();
    int num_labels = label_norm.size();
    results.reserve(num_pixels);

    try
    {
        GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
        NormDataFn toFn(estim_norm, label_norm);

        gc->setDataCost(& DataFn, & toFn);
        gc->setSmoothCost(& SmoothFn, & toFn);

        printf("\nBefore optimization energy is %f",gc->compute_energy());
        gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        printf("\nAfter optimization energy is %f",gc->compute_energy());

        for ( int  i = 0; i < num_pixels; i++ )
            results.push_back(gc->whatLabel(i));

        delete gc;

    }
    catch (GCException e){
        e.Report();
    }
}

bool VisualizeNorm(std::vector<cv::Vec3d>& norms, cv::Mat image)
{
    int width = image.cols;
    int height = image.rows;
    if (norms.size() < width * height)
    {
        std::cerr << "Normal vector is wrong.\n";
        return false;
    }

    cv::Vec3d light(0.5, 0, std::sqrt(3) / 2);
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int id = h * width + w;
            cv::Vec3d & norm_vector = norms[id];
            int color = (int)128 * (norm_vector[0] * light[0] + norm_vector[1] * light[1] + norm_vector[2] * light[2]);
            image.at<uchar>(h, w) = color;
        }
    }

    return true;

}