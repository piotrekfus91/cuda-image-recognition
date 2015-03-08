#ifndef INDEXEDMATWRAPPER_H_
#define INDEXEDMATWRAPPER_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common { namespace concurrency {

class IndexedMatWrapper {
public:
	IndexedMatWrapper();
	virtual ~IndexedMatWrapper();

	bool isPoison();
	void bePoison();

	int id;
	cir::common::MatWrapper matWrapper;
};

}}}
#endif /* INDEXEDMATWRAPPER_H_ */
