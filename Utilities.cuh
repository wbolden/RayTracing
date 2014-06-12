#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include "device_launch_parameters.h"
struct Sphere
{
	float3 position;
	uchar4 color;
	unsigned int texID;
	float radius;
	float reflectionWeight;
};

struct PointLight
{
	float3 position;
	uchar4 color;
	float intensity;
};

struct CameraInfo
{
	float3 position;
	float3 viewpos;
	float3 rotation;
};

inline __device__ uchar4 operator*(const float &b, const uchar4 &a)
{
	return make_uchar4(a.x*b, a.y*b, a.z*b, 0xFF);
}

inline __device__ uchar4 operator*(const uchar4 &a,const float &b)
{
	return make_uchar4(a.x*b, a.y*b, a.z*b, 0xFF);
}

inline __device__ uchar4 operator+(const uchar4 &a,const uchar4 &b)
{
	return make_uchar4(a.x+b.x, a.y+b.y, a.z+b.z, 0xFF);
}

inline __device__ uchar4 operator-(const uchar4 &a,const uchar4 &b)
{
	return make_uchar4(a.x-b.x, a.y-b.y, a.z-b.z, 0xFF);
}

inline __device__ void operator+=(uchar4 a,const uchar4 &b)
{
	a = a+b;
}

inline __device__ void operator-=(uchar4 a,const uchar4 &b)
{
	a = a-b;
}

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

inline __device__ float3 operator*(const float &b, const float3 &a)
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
	return rsqrtf(dot(a));
}

//Retruns the magnitude of a vector
inline __device__ float magnitude(const float3 &a)
{
	return sqrtf(dot(a));
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

inline __device__ float3 reflect(float3 incidentDirection, float3 normal)
{
	return -2*(dot(incidentDirection, normal)*normal) + incidentDirection;
}

//Creates a hexadecimal RGBA color from 
inline __device__ __host__ uchar4 rgb(int r, int g, int b)
{
	return make_uchar4(r, g, b, 0xFF);
}   

inline __device__ __host__ uchar4 rgb(int c)
{
	return make_uchar4(c, c, c, 0xFF);
}   

//Returns the lambert coefficient of two direction vectors
inline __device__ float lambert(float3 direction1, float3 direction2)
{
	return dot(direction1, direction2)/( magnitude(direction1) * magnitude(direction2) );
}

inline __device__ float frensel(float f0, float3 normal, float3 incidentDirection)
{
	float nv = 1 - dot(normal, -1*incidentDirection);

	return f0 + (1 - f0)*(nv * nv * nv * nv * nv); //F(0)  + (1 - F(0))(1 - (n*v))^5
}

inline __device__ float3 snell(float ni, float nt, float3 normal, float3 incidentDirection)
{
	float mu = ni/nt;
	float dotnv = dot(normal, -1*incidentDirection);

	return mu*incidentDirection - normal*(mu*dotnv + sqrtf(1-mu*mu*(1 - (dotnv*dotnv)))); //mu(v) - (mu(n * v) + sqrt(1 - mu^2 (1 - (n * v)^2)))
}

inline __device__ uchar4 saturate(uchar4 c1, uchar4 c2)
{
	return rgb((c1.x*c2.x)/0xFF, (c1.y*c2.y)/0xFF, (c1.z*c2.z)/0xFF);
}

inline __device__ float rand(unsigned int* seed)
{
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	unsigned long x = (unsigned long)*seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

	return ((float)x)/m;
}



#endif