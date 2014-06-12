#ifndef ACTION_CUH
#define ACTION_CUH
#include "Utilities.cuh"

class ActionHandler
{
public:
	ActionHandler(void);

	//temporary 
	ActionHandler(const Sphere& slist, const PointLight& pllist, Sphere* cuslist, PointLight* cupllist, int scout, int lcount);
	ActionHandler(Sphere* devslist, PointLight* devpllist, int* scout, int* lcount);

	void update(Sphere* slist, PointLight* pllist, int scount, int lcount);
};

#endif