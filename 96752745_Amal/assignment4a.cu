#include<iostream>
using namespace std;

__global__ void Max(int *d_out,int *d_a,int arraySize){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int s_a[1024];

     if(id < arraySize)
        s_a[tid] = d_a[id];
    __syncthreads();
    for(int s = 512; s>0; s = s/2)
    {
        __syncthreads();
        if(id>=arraySize || id+s>=arraySize)
            continue;
        if(tid<s)
            s_a[tid] = s_a[tid]>s_a[tid + s]?s_a[tid]:s_a[tid+s];
    }
    __syncthreads();
    if(tid==0)
        d_out[bid] = s_a[tid]; 
}

__global__ void Min(int *d_out,int *d_a,int arraySize){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int s_a[1024];

     if(id < arraySize)
        s_a[tid] = d_a[id];
    __syncthreads();
    for(int s = 512; s>0; s = s/2)
    {
        __syncthreads();
        if(id>=arraySize || id+s>=arraySize)
            continue;
        if(tid<s)
            s_a[tid] = s_a[tid]<s_a[tid + s]?s_a[tid]:s_a[tid+s];
    }
    __syncthreads();
    if(tid==0)
        d_out[bid] = s_a[tid]; 
}

int main()
{
	int arraySize,max=0,min;
	cout<<"Enter array size\n";
	cin>>arraySize;
	int h_a[arraySize],i,h_max,h_min;
	for(i=0;i<arraySize;i++)
	h_a[i]=5*i;
	min=h_a[0];
	for(i=0;i<arraySize;i++)
	{
		if(h_a[i]>max)
		max=h_a[i];
		if(h_a[i]<min)
		min=h_a[i];
	}

	int *d_a,*d_out1,*d_out2,*d_max,*d_min;
	cudaMalloc((void**)&d_a,arraySize*sizeof(int));
	cudaMalloc((void**)&d_out1, ceil(arraySize*1.0/1024)*sizeof(int));
	cudaMalloc((void**)&d_out2, ceil(arraySize*1.0/1024)*sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));
    cudaMalloc((void**)&d_min, sizeof(int));

    cudaMemcpy(d_a,h_a,arraySize*sizeof(int),cudaMemcpyHostToDevice);

    Max<<<ceil(arraySize*1.0/1024), 1024>>> (d_out1, d_a, arraySize);
    Max<<<1, 1024>>> (d_max, d_out1, ceil(arraySize*1.0/1024));
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    
    Min<<<ceil(arraySize*1.0/1024), 1024>>> (d_out2, d_a, arraySize);
    Min<<<1, 1024>>> (d_min, d_out2, ceil(arraySize*1.0/1024));
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    if(h_max==max)
    cout<<"Max element is "<<h_max<<endl;
    else
    cout<<"Some error has occured for calculating max!!"<<endl;

    if(h_min==min)
    cout<<"Min element is "<<h_min<<endl;
    else
    cout<<"Some error has occured for calculating min!!"<<endl;

    cudaFree(d_a);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_max);
    cudaFree(d_min);
}
