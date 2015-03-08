#ifndef INDEXEDMATWRAPPERBLOCKINGQUEUE_H_
#define INDEXEDMATWRAPPERBLOCKINGQUEUE_H_

#include "cir/common/concurrency/IndexedMatWrapper.h"
#include "cir/devenv/concurrency/BlockingQueue.h"

namespace cir { namespace common { namespace concurrency {

class IndexedMatWrapperBlockingQueue
	: public cir::devenv::concurrency::BlockingQueue<IndexedMatWrapper> {
public:
	IndexedMatWrapperBlockingQueue(int size);
	virtual ~IndexedMatWrapperBlockingQueue();
};

}}}
#endif /* INDEXEDMATWRAPPERBLOCKINGQUEUE_H_ */
