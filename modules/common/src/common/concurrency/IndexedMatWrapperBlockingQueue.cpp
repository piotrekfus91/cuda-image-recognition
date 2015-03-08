#include "cir/common/concurrency/IndexedMatWrapperBlockingQueue.h"

using namespace cir::devenv::concurrency;

namespace cir { namespace common { namespace concurrency {

IndexedMatWrapperBlockingQueue::IndexedMatWrapperBlockingQueue(int size)
	: BlockingQueue(size) {

}

IndexedMatWrapperBlockingQueue::~IndexedMatWrapperBlockingQueue() {

}

}}}
