#ifndef MATHOPS_CUH
#define MATHOPS_CUH

#include "device_launch_parameters.h"

inline __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline  __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ void operator-=(float3 &a, const float3 &b)
{
	a = a - b;
}

inline __device__ void operator+=(float3 &a, const float3 &b)
{
	a = a+b;
}

inline __device__ float3 operator/(const float3 &a, const float &b)
{
	return make_float3(a.x/b, a.y/b, a.z/b);
}

inline __device__ float3 operator*(const float3 &a, const float &b)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ void operator/=(float3 &a, const float &b)
{
	a = a/b;
}

inline __device__ void operator*=(float3 &a, const float &b)
{
	a = a*b;
}

//Returns the dot product of vectors a and b
inline __device__ float dot(const float3 &a, const float3 &b)
{
	return a.x*b.x+ a.y*b.y+ a.z*b.z;
}

//Returns the dot product of vectors a and a
inline __device__ float dot(const float3 &a)
{
	return a.x*a.x+ a.y*a.y+ a.z*a.z;
}

//Returns the inverse magnitude of a vector
inline __device__ float iMagnitude(const float3 &a)
{
	return rsqrt(dot(a));
}

//Retruns the magnitude of a vector
inline __device__ float magnitude(const float3 &a)
{
	return sqrt(dot(a));
}

//Returns a normalized vector
inline __device__ float3 normalize(const float3 &a)
{
	return a * iMagnitude(a);
}

//Returns the direction of a ray shot from its origin to a point
inline __device__ float3 getRayDirection(float3 origin, float3 point)
{
	return normalize(point - origin);
}

//Creates a hexadecimal RGBA color from 
inline __device__ int rgb(int r, int g, int b, float intensity)
{
	if(intensity >= 0)
	{
		return (((int)(r*intensity))<< 16) | (((int)(g*intensity))<< 8) | ((int)(b*intensity));
	}else{
		return 0;
	}
}   

inline __device__ int rgbBlend(int r0, int g0, int b0, float intensity0, float weight0, int r1, int g1, int b1, float intensity1)
{
	return rgb(r0, g0, b0, intensity0*weight0) + rgb(r1, g1, b1, intensity1 * (1-weight0));
}

//Returns the lambert coefficient of two direction vectors
inline __device__ float lambert(float3 direction1, float3 direction2)
{
	return dot(direction1, direction2)/( magnitude(direction1) * magnitude(direction2) );
}



#endif