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
    double lambda = 0.001;
    double sigma = 1;
    NormDataFn* norm_data = (NormDataFn*) data;
    cv::Vec3d label_norm1 = (norm_data->label_norm)[l1];
    cv::Vec3d label_norm2 = (norm_data->label_norm)[l2];
    return lambda * log(1 + cv::norm(label_norm1 - label_norm2) / (2 * pow(sigma, 2)));
}

void NormalOptimization(std::vector<cv::Vec3d> & estim_norm, std::vector<cv::Vec3d> & label_norm, std::vector<int> & label_id,
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
            results[i] = gc->whatLabel(i);

        delete gc;

    }
    catch (GCException e){
        e.Report();
    }


}