#include "Action.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>



__global__ void cuUpdate(Sphere* slist, PointLight* pllist, int scount, int lcount, float adx)
{

}




ActionHandler::ActionHandler()
{

}

ActionHandler::ActionHandler(const Sphere& slist, const PointLight& pllist, Sphere* cuslist, PointLight* cupllist, int scout, int lcount)
{

}

ActionHandler::ActionHandler(Sphere* devslist, PointLight* devpllist, int* scout, int* lcount)
{
	int tscount = 6;
	int lscount = 50;

	Sphere* slist = new Sphere[tscount];

	slist[0].position = make_float3(200, 0, 0);
	slist[0].radius = 30;
	slist[0].reflectionWeight = 0.6f;
	slist[0].color = rgb(0xFF);

	slist[1].position = make_float3(-20, -30, 0); 
	slist[1].radius = 15;
	slist[1].reflectionWeight = 0.4f;
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

	slist[5].position = make_float3(4, 20, -210);
	slist[5].radius = 9.9;
	slist[5].reflectionWeight = 0.0f;
	slist[5].color = rgb(0, 0, 0xFF);

	cudaMalloc((void**)&devslist, tscount * sizeof(Sphere));
	cudaMemcpy(devslist, slist, tscount * sizeof(Sphere), cudaMemcpyHostToDevice);

	PointLight* lpos = new PointLight[lscount];

	for(int i = 0; i < lscount; i++)
	{
		lpos[i].position = make_float3(500+i*2 * cosf(0+i), 1000, 500 +2*i * sinf(0+i));
		lpos[i].color = rgb(0xFF);
	}

	cudaMalloc((void**)&devpllist, lscount * sizeof(PointLight));
	cudaMemcpy(devpllist, lpos, lscount * sizeof(PointLight), cudaMemcpyHostToDevice);

}


float adx = 0;
void ActionHandler::update(Sphere* slist, PointLight* pllist, int scount, int lcount)
{
	adx += 0.0045f;


	int blockSize = 256;

	int gridSize = (scount + lcount + blockSize -1)/blockSize;

	cuUpdate<<<gridSize, gridSize>>>(slist, pllist, scount, lcount, adx);
}