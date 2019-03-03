#include <iostream>
#include <memory>
//#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace std;
using namespace cv;
bool findORBfeatures(cv::Mat img1, cv::Mat img2, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2);
// 相机内参
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
class Tracker
{
public:
    Tracker(Ptr<Feature2D> _detector, Ptr<cv::DescriptorMatcher> _matcher) :
        detector(_detector),
        matcher(_matcher)
    {}
    void setFirstFrame(const Mat frame);
    void process(const Mat frame);
    vector<cv::Point2f> matched1, matched2;
protected:
    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    vector<KeyPoint> first_kp;

};
void Tracker::setFirstFrame(const Mat frame)
{
    first_frame = frame.clone();
    detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);
}
void Tracker::process(const Mat frame)
{
    vector<KeyPoint> kp;
    Mat desc;
    detector->detectAndCompute(frame, noArray(), kp, desc);
    vector< vector<DMatch> > matches;
    matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx].pt);
            matched2.push_back(      kp[matches[i][0].trainIdx].pt);
        }
    }
}
class G2O_Optimizer
{
public:
    G2O_Optimizer(){
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
        solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);
        cam_params = new g2o::CameraParameters (fx, Eigen::Vector2d(cx, cy), 0.);
        cam_params->setId(0);
        optimizer.addParameter( cam_params );
    }//先设置两帧的pose，再用第一帧找空间点坐标，第一帧的边是pose和XYZ对应的uv，第二帧同理
    //void setVertex
    void setVertexSE3(g2o::SE3Quat pose){
        vertex_id = 0;
        for (int i=0;i<2;i++) {
            v_se3 = new g2o::VertexSE3Expmap();
            v_se3->setId(vertex_id);
            if ( vertex_id == 0) v_se3->setFixed( true );
            v_se3->setEstimate( pose );
            optimizer.addVertex( v_se3 );
            vertex_id++;
        }
    }
    void setVertexXYZ(vector<cv::Point2f> true_points,vector<cv::Point2f> true_points2){
        point_id=vertex_id;
        for (size_t i=0; i<true_points.size(); i++){
            v_p = new g2o::VertexSBAPointXYZ();
            v_p->setId(point_id);
            v_p->setMarginalized(true);
            double z = 1;
            double x = ( true_points[i].x - cx ) * z / fx;
            double y = ( true_points[i].y - cy ) * z / fy;
            v_p->setEstimate( Eigen::Vector3d(x,y,z) );
            optimizer.addVertex( v_p );
            point_id++;
            e = new g2o::EdgeProjectXYZ2UV();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>
            (optimizer.vertices().find(0)->second));
            e->setMeasurement(Eigen::Vector2d(true_points[i].x, true_points[i].y ) );
            e->information() = Eigen::Matrix2d::Identity();
            e->setRobustKernel(new g2o::RobustKernelHuber);
            e->setParameterId(0, 0);
            optimizer.addEdge(e);

            e = new g2o::EdgeProjectXYZ2UV();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>
            (optimizer.vertices().find(1)->second));
            e->setMeasurement(Eigen::Vector2d(true_points2[i].x, true_points2[i].y ) );
            e->information() = Eigen::Matrix2d::Identity();
            e->setRobustKernel(new g2o::RobustKernelHuber);
            e->setParameterId(0, 0);
            optimizer.addEdge(e);
        }

    }
    void startOptimize(){
        cout<<"开始优化"<<endl;
        optimizer.initializeOptimization();
        optimizer.setVerbose(true);
        optimizer.optimize(10);
        cout<<"变换矩阵为"<<endl;
        Eigen::Isometry3d pose=dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) )->estimate();
        cout<<"Pose="<<endl<<pose.matrix()<<endl;
        optimizer.save("ba.g2o");
    }



protected:
        g2o::SparseOptimizer    optimizer;
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        g2o::OptimizationAlgorithmLevenberg* solver;
        g2o::VertexSE3Expmap * v_se3;
        g2o::VertexSBAPointXYZ * v_p;
        g2o::CameraParameters * cam_params;
        g2o::EdgeProjectXYZ2UV * e;
        int vertex_id;
        int point_id;
};

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout<<"Usage: ba_example img1, img2"<<endl;
        exit(1);
    }

    // 读取图像
    cv::Mat img1 = cv::imread( argv[1] );
    cv::Mat img2 = cv::imread( argv[2] );

    Ptr<ORB> orb = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Tracker orb_tracker(orb, matcher);

    orb_tracker.setFirstFrame(img1);
    orb_tracker.process(img2);

    G2O_Optimizer demog2o;
    demog2o.setVertexSE3(g2o::SE3Quat());
    demog2o.setVertexXYZ(orb_tracker.matched1,orb_tracker.matched2);
    demog2o.startOptimize();

    return 0;
}

