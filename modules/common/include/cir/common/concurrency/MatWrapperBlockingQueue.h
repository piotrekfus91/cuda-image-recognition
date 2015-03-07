#ifndef MATWRAPPERBLOCKINGQUEUE_H_
#define MATWRAPPERBLOCKINGQUEUE_H_

#include "cir/common/MatWrapper.h"
#include "cir/devenv/concurrency/BlockingQueue.h"

namespace cir { namespace common { namespace concurrency {

class MatWrapperBlockingQueue
		: public cir::devenv::concurrency::BlockingQueue<cir::common::MatWrapper> {
public:
	MatWrapperBlockingQueue(int size);
	virtual ~MatWrapperBlockingQueue();
};

}}}
#endif /* MATWRAPPERBLOCKINGQUEUE_H_ */
