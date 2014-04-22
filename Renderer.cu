#include "Renderer.cuh"
#include "MathOps.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


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

struct Sphere
{
	float3 position;
	float radius;
	float reflectivity;

	__device__ float sintersection(float3 rayOrigin, float3 rayDirection, float distance)
	{
		return intersection(rayOrigin, rayDirection, distance, position, radius);
	}

	__device__ bool sshadowed(float3 rayOrigin, float3 rayDirection, float distance)
	{
		return shadowed(rayOrigin, rayDirection, distance, position, radius);
	}
};

struct CameraInfo
{
	float3 position;

	float3 viewpos;
	float3 rotation;
};

/*
	Just testing
*/
Sphere* slist;
Sphere* devslist;
int tcount = 13;

Renderer::Renderer(void)
{
	devPixels = nullptr;

	

	slist = new Sphere[tcount];

	slist[0].position = make_float3(0, 0, 0);
	slist[0].radius = 30;

	slist[1].position = make_float3(0, -60, 0);
	slist[1].radius = 10;

	slist[2].position = make_float3(10, -30, 0);
	slist[2].radius = 10;

	slist[3].position = make_float3(0, -40, 0);
	slist[3].radius = 5;

	slist[4].position = make_float3(10, 10, -30);
	slist[4].radius = 10;

	slist[5].position = make_float3(-0, 0, -27);
	slist[5].radius = 5;

	slist[6].position = make_float3(-0, 0, 10000);
	slist[6].radius = 9989;

	slist[7].position = make_float3(20, -30, 0);
	slist[7].radius = 8;

	slist[8].position = make_float3(0, -47, 0);
	slist[8].radius = 5;

	slist[9].position = make_float3(-10, 10, -30);
	slist[9].radius = 4;

	slist[10].position = make_float3(-10, -30, 0);
	slist[10].radius = 3;

	slist[11].position = make_float3(20, -40, 0);
	slist[11].radius = 5;

	slist[12].position = make_float3(15, 10, -30);
	slist[12].radius = 1;

	cudaMalloc((void**)&devslist, tcount * sizeof(Sphere));
	cudaMemcpy(devslist, slist, tcount * sizeof(Sphere), cudaMemcpyHostToDevice);
}

void Renderer::setResolution(int width, int height)
{
	if(devPixels == nullptr)
	{
		cudaMalloc((void**)&devPixels, width*height*sizeof(int));
	}
	else
	{
		cudaFree(devPixels);
		cudaMalloc((void**)&devPixels, width*height*sizeof(int));
	}
}

void Renderer::setProjectionMode(bool orthographic)
{
	this->orthographic = orthographic;
}


__device__ void castRay(float3 start, float3 direction, Sphere* slist, int scount, int& out, float adx)
{
	float3 lightPos = {0 +50*__cosf(adx), 50*__sinf(adx), -60};

	float d = -1;
	int snum = -1;

	for(int i = 0; i < scount; i++)
	{
		float td = slist[i].sintersection(start, direction, 100000);
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
			if(slist[i].sshadowed(contactPoint, cray, dot(contactPoint - lightPos))	) //add ray length parameter, for shadows ray length is distance to light
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


__global__ void render(int width, int height, float3 cameraPos, Sphere* slist, int scount, int* out, float adx, float ady)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x < width && y < height)
	{
		int i= y * width + x;

		/*
			Normalize the values.
			The smaller of the two (width or height) is normalized to a range [-1, 1]
			The larger of the two is normalized depending on the value of the smaller

			Both are normalized to the range [-1, 1] if they are the same
		*/

		float norx;
		float nory;

		if(width < height)
		{
			norx = (x - 0.5*width)/(0.5*width);
			nory = -(y - 0.5*height)/(0.5*width);
		}
		else
		{
			norx = (x - 0.5*width)/(0.5*height);
			nory = -(y - 0.5*height)/(0.5*height);
		}

		

		out[i] = 0;

		float3 dir = {norx, nory, 1};	//Screen location

		dir = getRayDirection(cameraPos, dir);

		float3 pos = {-0, -10, -70};

		castRay(cameraPos + pos, dir, slist, scount, out[i], adx);
	}
}



float adx = 0.00;
float ady = 0.00;

void Renderer::renderFrame(int width, int height, int* pixels)
{
	int numElements = width*height;
	int size = sizeof(pixels[0]) * numElements;

	 adx += 0.003f;
	 ady += 0.003f;

	float3 cameraPos = {0, 0, 0};


	dim3 blockSize(16, 16); //16 * 16 threads per block

	int xGridSize = (width + blockSize.x-1)/blockSize.x; 
	int yGridSize = (height + blockSize.y-1)/blockSize.y;

	dim3 gridSize( xGridSize, yGridSize);


	render<<< gridSize, blockSize >>>(width, height, cameraPos, devslist, tcount, devPixels,  adx,  ady);

	cudaMemcpy(pixels, devPixels, size, cudaMemcpyDeviceToHost);
}


Renderer::~Renderer(void)
{
	cudaFree(devPixels);
}
