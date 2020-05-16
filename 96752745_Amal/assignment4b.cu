#include<iostream>
using namespace std;

__global__ void Sum(int *d_out,int *d_a,int arraySize){
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
            s_a[tid] += s_a[tid+s];
    }
    __syncthreads();
    if(tid==0)
        d_out[bid] = s_a[tid]; 
}

__global__ void Prod(int *d_a,int *d_b,int *d_p,int arraySize){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < arraySize)
	d_p[id] = d_a[id]*d_b[id];	
}

int main()
{
	int arraySize;
	cout<<"Enter array size\n";
	cin>>arraySize;
	int h_a[arraySize],h_b[arraySize],i,h_sum;

	for(i=0;i<arraySize;i++)
	{
		h_a[i]=2*i;
		h_b[i]=3*i;
	}
	int *d_a,*d_b,*d_out,*d_sum,*d_p;
	cudaMalloc((void**)&d_a, sizeof(int)*arraySize);
	cudaMalloc((void**)&d_b, sizeof(int)*arraySize);
	cudaMalloc((void**)&d_p, sizeof(int)*arraySize);
	cudaMalloc((void**)&d_out, ceil(1.0*arraySize/1024)*sizeof(int));
	cudaMalloc((void**)&d_sum,sizeof(int));

	cudaMemcpy(d_a,h_a,arraySize*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,arraySize*sizeof(int),cudaMemcpyHostToDevice);
	Prod<<<ceil(1.0*arraySize/1024),1024>>>(d_a,d_b,d_p,arraySize);

	Sum<<<ceil(arraySize*1.0/1024), 1024>>> (d_out, d_p, arraySize);
    Sum<<<1, 1024>>> (d_sum, d_out, ceil(arraySize*1.0/1024));
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<"Dot Product is "<<h_sum<<endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_p);
    cudaFree(d_out);
    cudaFree(d_sum);
}
