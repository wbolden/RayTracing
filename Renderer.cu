#include "Renderer.cuh"
#include "MathOps.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>






__device__ void shadow()
{

}

__device__ void refraction()
{

}

__device__ void reflection()
{

}


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
	float3 lightPos = {0 +200*__cosf(adx), 300+ 200*__cosf(adx), 200*__sinf(adx)};


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

__global__ void cuRender(float3* rayDirections, int width, int height, float3 cameraPos, Sphere* slist, int scount, cudaSurfaceObject_t out, float adx, float ady)
{
	extern __shared__ Sphere sharedslist[];

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

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

		if(width < height)
		{
			normalizedX = (x - 0.5*width)/(0.5*width);
			normalizedY = (y - 0.5*height)/(0.5*width);
		}
		else
		{
			normalizedX = (x - 0.5*width)/(0.5*height);
			normalizedY = (y - 0.5*height)/(0.5*height);
		}

		float3 lensLocation;
		float3 origin = {0, 0, 0};

		lensLocation = make_float3(normalizedX, normalizedY, 1);

		float3 dir = getRayDirection(origin, lensLocation);


	//	int aamt = 5;
#define aamt 7
#define offs aamt/2


		float norx[aamt][aamt];
		float nory[aamt][aamt];

		for(int ix = 0; ix < aamt; ix++)
		{
			for(int iy = 0; iy < aamt; iy++)
			{
				float aix = ix - offs;;
				float aiy = iy - offs;

				normalizedX = (x+aix/(1.5*offs) - 0.5*width)/(0.5*height);
				normalizedY = (y+aiy/(1.5*offs)  - 0.5*height)/(0.5*height);

				norx[ix][iy] = normalizedX;
				nory[ix][iy] = normalizedY;

			}

		}

		

		
		unsigned int ots[aamt][aamt];
		int dirs[aamt][aamt];

		float3 pos = {-0, 20, -100};

		for(int ix = 0; ix < aamt; ix++)
		{
			for(int iy = 0; iy < aamt; iy++)
			{

				lensLocation = make_float3(norx[ix][iy], nory[ix][iy], 1);

				dir = getRayDirection(origin, lensLocation);

				castRay(cameraPos + pos, dir, sharedslist, scount, ots[ix][iy], adx);

			}

		}
		
		int rr = 0;
	//	int rg = 0;
	//	int rb = 0;

		for(int ix = 0; ix < aamt; ix++)
		{
			for(int iy = 0; iy < aamt; iy++)
			{



				rr += (ots[ix][iy]<<24) >> 24;

			}

		}

		rr /= aamt*aamt;

		result = rgb(rr, rr, rr, 1);



	//	float3 pos = {-0, 20, -100};

	//	castRay(cameraPos + pos, dir, sharedslist, scount, result, adx);


		surf2Dwrite(result, out, x * sizeof(unsigned int), y, cudaBoundaryModeClamp);
	}
}

__global__ void cuCreateLens(float3 *rayDirections, int width, int height, bool sphere, float radius)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x < width && y < height)
	{
		float normalizedX;
		float normalizedY;

		if(width < height)
		{
			normalizedX = (x - 0.5*width)/(0.5*width);
			normalizedY = (y - 0.5*height)/(0.5*width);
		}
		else
		{
			normalizedX = (x - 0.5*width)/(0.5*height);
			normalizedY = (y - 0.5*height)/(0.5*height);
		}

		float3 lensLocation;
		float3 origin = {0, 0, 0};

		if(sphere)
		{
			float sphereZCoord = sqrt(radius*radius - normalizedX*normalizedX - normalizedY*normalizedY);
			lensLocation = make_float3(normalizedX, normalizedY, sphereZCoord);
			
			rayDirections[y * width + x] = getRayDirection(origin, lensLocation);
		}
		else
		{
			lensLocation = make_float3(normalizedX, normalizedY, 1);

			rayDirections[y * width + x] = getRayDirection(origin, lensLocation);
		}
	}
}

#include <stdlib.h>


/*
	Just testing
*/
Sphere* slist;
Sphere* devslist;
int tcount = 6;

Renderer::Renderer(void)
{
	viewLens = nullptr;

	

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

dim3 blockSize;
dim3 gridSize;

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

void Renderer::createLens(int width, int height)
{
	if(viewLens == nullptr)
	{
		cudaMalloc((void**)&viewLens, renderWidth * renderHeight * sizeof(float3));
	}
	
	cuCreateLens<<<gridSize, blockSize>>>(viewLens, renderWidth, renderHeight, false, 0);
}

void Renderer::createSphericalLens(int width, int height, int radius)
{
	if(viewLens == nullptr)
	{
		cudaMalloc((void**)&viewLens, renderWidth * renderHeight * sizeof(float3));
	}

	cuCreateLens<<<gridSize, blockSize>>>(viewLens, renderWidth, renderHeight, true, 4);
}











float adx = 0.00;
float ady = 0.00;


void Renderer::renderFrame(cudaSurfaceObject_t pixels)
{
	 adx += 0.003f;
	 ady += 0.003f;

	float3 cameraPos = {0, 0, 0};


	unsigned int smem = sizeof(Sphere)*tcount;

	cuRender<<<gridSize, blockSize, smem >>>(viewLens ,renderWidth, renderHeight, cameraPos, devslist, tcount, pixels,  adx,  ady);
}



Renderer::~Renderer(void)
{
	
}
