#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

struct matrix{
  int m;
  int n;
  float* data;
};
void matmul(matrix* a,matrix* b,matrix* c){
  assert(a->n==b->m&&a->m==c->m&&b->n==c->n);
  for(int i=0;i<c->m;++i){
    for(int j=0;j<c->n;++j){
        float ans=0;
        for(int k=0;k<a->n;++k){
            ans+=a->data[i*a->n+k]*b->data[k*b->n+j];
        }
        c->data[i*c->n+j]=ans;
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    float* prob_data=new float[batch*k];
    matrix softmax_prob=matrix{batch,k,prob_data};
    matrix mat_theta=matrix{n,k,theta};
    int num_examples=m;
    for(int ii=0;ii<num_examples;ii+=batch){
        matrix X_batch=matrix{batch,n,(float*)X+ii*n};
        unsigned char* y_batch=(unsigned char*)y+ii;
        matmul(&X_batch,&mat_theta,&softmax_prob);
        for(int j=0;j<softmax_prob.m;++j){
            float sum=0;
            for(int p=0;p<softmax_prob.n;++p){
                softmax_prob.data[j*softmax_prob.n+p]=exp(softmax_prob.data[j*softmax_prob.n+p]);
                sum+=softmax_prob.data[j*softmax_prob.n+p];
            }
            for(int p=0;p<softmax_prob.n;++p){
                softmax_prob.data[j*softmax_prob.n+p]/=sum;
            }
        }

        for(int i=0;i<n;++i){
            for(int j=0;j<k;++j){
                float grad_elem=0;
                for(int p=0;p<batch;++p){
                    grad_elem+=X_batch.data[p*X_batch.n+i]*(softmax_prob.data[p*softmax_prob.n+j]-(y_batch[p]==j));
                }
                theta[i*k+j]-=lr*grad_elem/batch;
            }
        }
    }
    delete[] prob_data;
  
    
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}