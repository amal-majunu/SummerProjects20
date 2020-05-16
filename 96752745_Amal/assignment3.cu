#include<iostream>
using namespace std;

__global__ void Transpose(int *d_a,int r,int c){

int i = blockIdx.x*blockDim.x+threadIdx.x;
int j = blockIdx.y*blockDim.y+threadIdx.y;

__syncthreads();

if(i<c && j<r)
{
    int id1 = i+j*c;
    int id2 = j+i*r;
	int t = d_a[id1];
	__syncthreads();
	d_a[id2]=t;
	
}
	
}

int main()
{
	int r,c,i,j;
	cout<<"Enter the number of rows and columns:\n";
	cin>>r>>c;
	int h_a[r][c]={0},h_b[c][r];
	for(i=0;i<r;i++)
	{
		for(j=0;j<c;j++)
		h_a[i][j]=2*i+j;
	}
	cout<<"Given array is:\n";
	for(i=0;i<r;i++)
	{
		for(j=0;j<c;j++)
		cout<<h_a[i][j]<<" ";
		cout<<"\n";
	}
	int *d_a;
	cudaMalloc((void**)&d_a, r*c*sizeof(int));

	cudaMemcpy(d_a, h_a, r*c*sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil(1.0*c/dimBlock.x), (int)ceil(1.0*r/dimBlock.y));
	Transpose<<<dimGrid,dimBlock>>>(d_a,r,c);
	cudaMemcpy(h_b, d_a, r*c*sizeof(int), cudaMemcpyDeviceToHost);
	cout<<"The transpose matrix is:\n";
	for(i=0;i<c;i++)
	{
		for(j=0;j<r;j++)
		cout<<h_b[i][j]<<" ";
		cout<<"\n";
	}

	cudaFree(d_a);
	return 0;
}
