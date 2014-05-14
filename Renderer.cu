#include "Renderer.cuh"
#include "MathOps.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>






__device__ void cuLight()
{

}	

__device__ void cuRefraction()
{

}

__device__ void cuReflection()
{

}

__device__ void cuIntersection()
{

}

__device__ void cuCastRay()
{

}


//To be moved into cu... methods ___
__device__ float intersection(float3 rayOrigin, float3 rayDirection, float distance,float3 position, float radius)
{
	float3 dist = position - rayOrigin;

	float b = dot(rayDirection, dist);

	float d = b*b - dot(dist, dist) + radius * radius;

	if(d < 0 )	//If the object is behind the ray, return
	{
		return -1;
	}


	float t0 = b -sqrt(d);
	float t1 = b +sqrt(d);

	float t = distance;


	if(t0 > 0.1 && t0 < t)
	{
		t = t0;
	}

	if(t1 > 0.1 && t1 < t)
	{
		t = t1;
	}

	return t;
}

__device__ bool shadowed(float3 rayOrigin, float3 rayDirection, float distance,float3 position, float radius)
{
	float3 dist = position - rayOrigin;

	float ddist = dot(dist, dist);

	if(ddist > distance)
	{
		return true;
	}

	float b = dot(rayDirection, dist);

	float d = b*b - ddist + radius * radius;

	if(d < 0 )	//If the object is behind the ray, return
	{
		return true;
	}



	float t0 = b -sqrt(d);
	float t1 = b +sqrt(d);



	float t = distance*distance;

	if(t0 > 0.1 && t0 < t)
	{
		t = t0;
	}

	if(t1 > 0.1 && t1 < t)
	{
		t = t1;
	}

	if(t >= distance*distance)
	{
		return true;
	} 
	return false;
}

__device__ void castRay(float3 start, float3 direction, Sphere* slist, int scount, unsigned int& out, float adx)
{
	float3 lightPos = {0 +200*__cosf(adx), 300+ 200*__cosf(adx), 300*__cosf(adx)};


	float d = -1;
	int snum = -1;

	for(int i = 0; i < scount; i++)
	{
		float td = intersection(start, direction, 100000, slist[i].position, slist[i].radius);
		if((d < 0 || td < d) && td > 0)
		{
			d = td;
			snum = i;
		}
		
	}

	if(d > 0)
	{
		float3 contactPoint = start +direction*d;

		float3 cray = getRayDirection(contactPoint, lightPos);

		float3 snorm = getRayDirection(slist[snum].position, contactPoint);

		float intensity = lambert(cray, snorm);	//Compute lambert intensity

		for(int i = 0; i < scount; i++)
		{												
			if(shadowed(contactPoint, cray, dot(contactPoint - lightPos), slist[i].position, slist[i].radius	)) //add ray length parameter, for shadows ray length is distance to light
			{
				out = rgb(0xFF, 0xFF, 0xFF, intensity);	//Color in light
			}
			else
			{
				out = 0x000000;		//Color in shadow
				break;
			}
		}

	}
	else
	{
		out = 0x222222;		//Background color
	}
}
//To be moved into cu... methods ^^^

__global__ void cuRender(cudaSurfaceObject_t out, int width, int height, int aa, float3 cameraPos, Sphere* slist, int scount, float adx, float ady)
{
	extern __shared__ Sphere sharedslist[];

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(blockIdx.x *blockIdx.y ==1)
	{
		slist[0].position.x+=0.09;
	}

	__syncthreads();

	if(x < width && y < height)
	{
	
		unsigned int index = y * width + x;
		unsigned int tindex = threadIdx.x*blockDim.y +threadIdx.y;

		if(tindex < scount)
		{
		//	sharedslist[tindex] = slist[tindex];

			sharedslist[tindex].position = slist[tindex].position;
			sharedslist[tindex].radius = slist[tindex].radius;
		}
		__syncthreads();

		unsigned int result = 0;


		


		float normalizedX;
		float normalizedY;



		float3 lensLocation;
		float3 dir;
		float3 origin = {0, 0, 0};

		 
		int offs = aa/2;

		int rr = 0;
		float3 pos = {-0, 20, -100- adx*20};

		for(int ix = 0; ix < aa; ix++)
		{
			for(int iy = 0; iy < aa; iy++)
			{
				float aix = ix - offs;;
				float aiy = iy - offs;

				normalizedX = (x+aix/(1.5*offs) - 0.5*width)/(0.5*height);
				normalizedY = (y+aiy/(1.5*offs)  - 0.5*height)/(0.5*height);

				lensLocation = make_float3(normalizedX, normalizedY, 1);

				dir = getRayDirection(origin, lensLocation);

				castRay(cameraPos + pos, dir, sharedslist, scount, result, adx);

				rr += (result<<24) >> 24;

			}

		}

		rr /= aa*aa;

		result = rgb(rr, rr, rr, 1);


		surf2Dwrite(result, out, x * sizeof(unsigned int), y, cudaBoundaryModeClamp);
	}
}







//To be cleaned up 

float adx = 0.00;		//remove
float ady = 0.00;		//remove
Sphere* slist;
Sphere* devslist;
int tcount = 6;

Renderer::Renderer(void)
{

	slist = new Sphere[tcount];

	slist[0].position = make_float3(0, 0, 0);
	slist[0].radius = 30;




	slist[1].position = make_float3(0, -60, 0);
	slist[1].radius = 10;

	slist[2].position = make_float3(10, -30, 0);
	slist[2].radius = 10;

	slist[3].position = make_float3(0, -40, 0);
	slist[3].radius = 5;

	slist[4].position = make_float3(-0, -10000, 0);
	slist[4].radius = 9960;

	slist[5].position = make_float3(20, -30, 0);
	slist[5].radius = 8;



	cudaMalloc((void**)&devslist, tcount * sizeof(Sphere));
	cudaMemcpy(devslist, slist, tcount * sizeof(Sphere), cudaMemcpyHostToDevice);
}

void Renderer::setResolution(int width, int height)
{
	blockSize = dim3(16,16); //16 * 16 threads per block

	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	gridSize = dim3( xGridSize, yGridSize);

	renderWidth = width;
	renderHeight = height;
}

void Renderer::setProjectionMode(bool orthographic)
{
	this->orthographic = orthographic;
}

void Renderer::renderFrame(cudaSurfaceObject_t pixels)
{
	 adx += 0.003f;
	 ady += 0.003f;
	float3 cameraPos = {0, 0, 0};		//put in scene

	unsigned int smem = sizeof(Sphere)*tcount;		//to be replaced
	cuRender<<<gridSize, blockSize, smem >>>(pixels, renderWidth, renderHeight, 3, cameraPos, devslist, tcount, adx,  ady);
}

Renderer::~Renderer(void)
{
	delete[] slist;
	cudaFree(devslist);
}
