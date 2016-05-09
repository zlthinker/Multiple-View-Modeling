#include <iostream>
#include "GCO/energy.h"
#include "GCO/GCoptimization.h"
#include <opencv/highgui.h>
#include <fstream>
#include <sstream>

using namespace std;

void NormalOptimization(std::vector<cv::Vec3d> & estim_norm, std::vector<cv::Vec3d> & label_norm,
                        int width, int height, std::vector<int> & results);
int GCO_main(int argc, char **argv);
bool VisualizeNorm(std::vector<cv::Vec3d>& norms, cv::Mat image);

int main() {

    int width, height;
    int num_pixels;
    int num_labels;
    std::vector<cv::Vec3d> estim_norms;
    std::vector<cv::Vec3d> label_norms;
    std::vector<cv::Vec3d> opti_norms;
    std::vector<int> results;


    // read estimated normals
    std::string estim_norm_file = "/Users/Larry/HKUST/Lessons/COMP5421_Computer_Vision/project_4/data/norm/initalNorm9.txt";
    std::ifstream f_norm(estim_norm_file);
    if (f_norm.is_open())
    {
        f_norm >> width >> height;
    }

    num_pixels = width * height;
    estim_norms.reserve(num_pixels);
    opti_norms.reserve(num_pixels);
    results.reserve(num_pixels);

    while (f_norm.is_open() && !f_norm.eof())
    {
        double x, y, z;
        f_norm >> x >> y >> z;
        estim_norms.push_back(cv::Vec3d(x, y, z));
    }
    f_norm.close();
    std::cout << "Reading estimated normal file finished.\n";
    estim_norms.resize(num_pixels);

    // read norm label
    std::string label_norm_file = "/Users/Larry/HKUST/Lessons/COMP5421_Computer_Vision/project_4/data/semi-sphere.txt";
    std::ifstream f_label(label_norm_file);
    while (f_label.is_open() && !f_label.eof())
    {
        double x, y, z;
        f_label >> x >> y >> z;
        label_norms.push_back(cv::Vec3d(x, y, z));
    }
    f_label.close();
    std::cout << "Reading label normal file finished.\n";
    label_norms.resize(label_norms.size() - 1);
    num_labels = label_norms.size();

    // before optimization
    cv::Mat init_image(height, width, CV_8UC1);
    if (!VisualizeNorm(estim_norms, init_image))
    {
        std::cerr << "Fail to visualize normal vectors.\n";
        return -1;
    }
    std::string image_before = "/Users/Larry/HKUST/Lessons/COMP5421_Computer_Vision/project_4/data/norm/init_image8.png";
    cv::imwrite(image_before, init_image);

//    cv::namedWindow("Before optimization", cv::WINDOW_NORMAL);
//    cv::imshow("Before optimization", init_image);

    NormalOptimization(estim_norms, label_norms, width, height, results);

    // save optimized normal
    std::string opti_norm_file = "/Users/Larry/HKUST/Lessons/COMP5421_Computer_Vision/project_4/data/norm/optimalNormal8.txt";
    std::ofstream f_opti(opti_norm_file);

    if (!f_opti.is_open())
    {
        std::cout << "Fail to open " << opti_norm_file << '\n';
        return -1;
    }
    f_opti << width << ' ' << height << '\n';
    for (int i = 0; i < results.size(); i++)
    {
        cv::Vec3d & opti_vector = label_norms[ results[i] ];
        opti_norms.push_back(opti_vector);
        f_opti << opti_vector[0] << ' ' << opti_vector[1] << ' ' << opti_vector[2] << '\n';
    }

    // after optimization
    cv::Mat opti_image(height, width, CV_8UC1);
    if (!VisualizeNorm(opti_norms, opti_image))
    {
        std::cerr << "Fail to visualize normal vectors.\n";
        return -1;
    }
//    cv::imshow("After optimization", opti_image);
    std::string image_after = "/Users/Larry/HKUST/Lessons/COMP5421_Computer_Vision/project_4/data/norm/opti_image8.png";
    cv::imwrite(image_after, opti_image);


    return 0;
}