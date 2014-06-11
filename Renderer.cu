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

__device__ unsigned int cuCastRay()
{
	return 0;
}


//To be moved into cu... methods ___
__device__ float intersection(float3 rayOrigin, float3 rayDirection, float3 position, float radius)
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

	if(t0 < 0)
	{
		return t1;
	}
	else
	{
		return t0;
	}
}

__device__ bool shadowed(float3 rayOrigin, float3 rayDirection, float distanceToLight,float3 position, float radius)
{
	float3 distance = position - rayOrigin;

	float dotDistance = dot(distance, distance);

	if(dotDistance > distanceToLight)
	{
		return false;
	}

	float b = dot(rayDirection, distance);

	float d = b*b - dotDistance + radius * radius;

	if(d < 0 )	//If the object is behind the ray, return
	{
		return false;
	}
	
	float t = b -sqrtf(d); 

	if(t < 0)
	{
		t =  b +sqrtf(d);
		if(t < 0)
		{
			return false;
		}
	}

	return true;
}

__device__ contactInfo castRay(float3 start, float3 direction, Sphere* slist, int scount, float adx)
{
	contactInfo rval;
	rval.reflect = false;

	


	float3 lightPos = {0 +100*__cosf(adx), 100, 100*__sinf(adx)};

//	float3 lightPos = {0, 100, -200};





	////////////////
	rval.reflectionWeight = 0.6f;
	////////////////





	float d = -1;
	int snum = -1;

	for(int i = 0; i < scount; i++)
	{
		float td = intersection(start, direction, slist[i].position, slist[i].radius);
		if((d < 0 || td < d) && td > 0)
		{
			d = td;
			snum = i;
		}
		
	}

	d-= 0.05f; //Accounts for floating point errors (prevents the contact point from being inside an object)

	if(snum > -1)
	{
		float3 contactPoint = start +direction*d;

		float3 cray = getRayDirection(contactPoint, lightPos);

		float3 snorm = getRayDirection(slist[snum].position, contactPoint);

		//////
		rval.startPosition = contactPoint;
		rval.normal = snorm;
		rval.reflectionWeight = slist[snum].reflectionWeight;

		if(!(rval.reflectionWeight == 0))
		{
			rval.reflectionWeight = frensel(slist[snum].reflectionWeight, snorm, direction);
		}
		///////




		float intensity = lambert(cray, snorm);	//Compute lambert intensity

		for(int i = 0; i < scount; i++)
		{												
			if(!shadowed(contactPoint, cray, dot(contactPoint - lightPos), slist[i].position, slist[i].radius)) //add ray length parameter, for shadows ray length is distance to light
			{
				if(intensity < 0.07)
				{
					intensity = 0.07;
				}

				rval.basicColor = slist[snum].color * intensity;;
				rval.reflect = true;
			}
			else
			{
				rval.basicColor = slist[snum].color * 0.07f;
				rval.reflect = true;

				break;
			}
		}




	}
	else
	{
		rval.basicColor = rgb(0, 0, 0);		//Background color //0x222222
		//rval.basicColor = rgb(126,192,238);


		rval.reflect = false;
	}


	return rval;
}
//To be moved into cu... methods ^^^



//ADD FRENSEL
__device__ uchar4 cuTraceRay(float3 startPosition, float3 startDirection, Sphere* sslist, int scount,float adx)
{
	float3 dir = startDirection;
	contactInfo inft;
	inft = castRay(startPosition, dir, sslist, scount, adx);

	if(true)
	{
		#define NUM 5

		uchar4 rcols[NUM];
		float rw[NUM];

		int i = 0;

		while(i < NUM)
		{
			rcols[i] = inft.basicColor;
			rw[i] = inft.reflectionWeight;

			if(inft.reflect)
			{
				dir = reflect(dir, inft.normal);
				normalize(dir);

				inft = castRay(inft.startPosition, dir, sslist, scount, adx);
			}
			else
			{
				break;
			}

			i++;
		}

		while(i >0)
		{
			rcols[i-1] = rcols[i-1]*(1-rw[i-1]) + rcols[i] * rw[i-1];
			i--;
		}

		return rcols[0];
	}
	else
	{
		return inft.basicColor;
	}
}

__device__ uchar4 cuRenderPixelAA(int pixelX, int pixelY, int width, int height, int aa,float3 position, Sphere* sharedslist, int scount, float adx) //adx to be removed
{
	uchar4 result;


	float offs = 0;

	if(aa % 2 == 0)
	{
		 offs = aa;//finish
	}
	else
	{
		 offs = (int)aa/2;
	}
	 
	int r = 0;
	int g = 0;
	int b = 0;
	for(int ix = 0; ix < aa; ix++)
	{
		for(int iy = 0; iy < aa; iy++)
		{
			float aix = ix - offs;
			float aiy = iy - offs;

			float normalizedX = (pixelX+aix/(1.5*offs) - 0.5*width)/(0.5*height);
			float normalizedY = (pixelY+aiy/(1.5*offs)  - 0.5*height)/(0.5*height);

			float3 lensLocation = {normalizedX, normalizedY, 1.5};

			float3 dir = getRayDirection(make_float3(0, 0, 0), lensLocation);

			result = cuTraceRay(position, dir, sharedslist, scount, adx);

			r += result.x;
			g += result.y;
			b += result.z;

		}
	}
	r /= aa*aa;
	g /= aa*aa;
	b /= aa*aa;
	result = rgb(r,g, b);
	return result;
}

__device__ uchar4 cuRenderPixel(int pixelX, int pixelY, int width, int height, float3 position, Sphere* sharedslist, int scount, float adx) //adx to be removed
{
	float normalizedX = (pixelX - 0.5*width)/(0.5*height);
	float normalizedY = (pixelY - 0.5*height)/(0.5*height);

	float3 lensLocation = {normalizedX, normalizedY, 1.5};

/*
	int oldx = lensLocation.x;
	int oldz = lensLocation.z;
	int rrot = adx*10;
	lensLocation.x = oldx * cosf(rrot) - oldz * sinf(rrot);
	lensLocation.z = oldz * sinf(rrot) + oldx * cosf(rrot);
*/

	uchar4 result;
	float3 dir = getRayDirection(make_float3(0, 0, 0), lensLocation);

	result = cuTraceRay(position, dir, sharedslist, scount, adx);

	return result;
}

__global__ void cuRender(cudaSurfaceObject_t out, int width, int height, int aa, float3 cameraPos, Sphere* slist, int scount, float adx, float ady, int renderOffset)
{ 
	extern __shared__ Sphere sharedslist[];

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(blockIdx.x *blockIdx.y ==1)
	{
		slist[0].position.x = __sinf(adx) * 100;
		slist[0].position.z = __cosf(adx) * 100;



		//slist[2].position.x += 0.01;

		slist[6].position.z += __sinf(adx);

		//slist[6].radius += 0.5;
	}


	__syncthreads();

	if(x < width && y < height)
	{
		unsigned int tindex = threadIdx.x*blockDim.y +threadIdx.y;

		if(tindex < scount)
		{
			sharedslist[tindex].position = slist[tindex].position;
			sharedslist[tindex].radius = slist[tindex].radius;
			sharedslist[tindex].reflectionWeight = slist[tindex].reflectionWeight;
			sharedslist[tindex].color = slist[tindex].color;
		}
		__syncthreads();

		uchar4 result;

		if(aa > 1)
		{
			result = cuRenderPixelAA(x, y, width, height, aa, cameraPos, sharedslist, scount, adx);
		}
		else
		{
			result = cuRenderPixel(x, y, width, height, cameraPos, sharedslist, scount, adx);
		}

		surf2Dwrite(result, out, renderOffset* sizeof(uchar4)+ x * sizeof(uchar4), y, cudaBoundaryModeClamp);
	}
}







//To be cleaned up 

float adx = 0.00;		//remove
float ady = 0.00;		//remove
Sphere* slist;
Sphere* devslist;
int tcount = 7;

Renderer::Renderer(void)
{

	slist = new Sphere[tcount];

	slist[0].position = make_float3(200, 0, 0);
	slist[0].radius = 30;
	slist[0].reflectionWeight = 0.1f;
	slist[0].color = rgb(0xFF);

	slist[1].position = make_float3(0, -0, 0);
	slist[1].radius = 15;
	slist[1].reflectionWeight = 0.0f;
	slist[1].color = rgb(0xFF, 0, 0);

	slist[2].position = make_float3(10, -30, 0);
	slist[2].radius = 10;
	slist[2].reflectionWeight = 0.6f;
	slist[2].color = rgb(0, 0xFF, 0);

	slist[3].position = make_float3(0, -40, 0);
	slist[3].radius = 5;
	slist[3].reflectionWeight = 0.0f;
	slist[3].color = rgb(0xFF);

	slist[4].position = make_float3(-0, -10000, 0);
	slist[4].radius = 9960;
	slist[4].reflectionWeight = 0.0f;
	slist[4].color = rgb(0xFF);
	
//	slist[5].position = make_float3(20, -30, 0);
//	slist[5].radius = 8;

	slist[5].position = make_float3(-0, 20, -210);
	slist[5].radius = 8;
	slist[5].reflectionWeight = 0.0f;
	slist[5].color = rgb(0, 0, 0xFF);

//	slist[6].position = make_float3(20, 500, 1000);
//	slist[6].radius = 800;

	slist[6].position = make_float3(0, 20000, -230); //20
	slist[6].radius = 1000;
	slist[6].reflectionWeight = 0.0f;
	slist[6].color = rgb(0xFF);

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

int ad = 0;
int aa = 1;

void Renderer::renderFrame(cudaSurfaceObject_t pixels)
{
	 adx += 0.0045f;
	// ady += 0.003f;
	float3 cameraPos = {0, 20, -200};		//put in scene

	 
	unsigned int smem = sizeof(Sphere)*tcount;		//to be replaced


	if(true) //3d
	{
		cameraPos.x -= 0.5f;
		cuRender<<<gridSize, blockSize, smem >>>(pixels, renderWidth/2, renderHeight, aa, cameraPos, devslist, tcount, adx,  ady, 0);

		cameraPos.x += 1;
		cuRender<<<gridSize, blockSize, smem >>>(pixels, renderWidth/2, renderHeight, aa, cameraPos, devslist, tcount, adx,  ady, renderWidth/2);
	}
	else
	{
		cuRender<<<gridSize, blockSize, smem >>>(pixels, renderWidth, renderHeight, aa, cameraPos, devslist, tcount, adx,  ady, 0);
	}
}

Renderer::~Renderer(void)
{
	delete[] slist;
	cudaFree(devslist);
}
