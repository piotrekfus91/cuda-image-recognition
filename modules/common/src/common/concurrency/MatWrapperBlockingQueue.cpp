#include "cir/common/concurrency/MatWrapperBlockingQueue.h"

using namespace cir::devenv::concurrency;

namespace cir { namespace common { namespace concurrency {

MatWrapperBlockingQueue::MatWrapperBlockingQueue(int size)
		: BlockingQueue(size) {

}

MatWrapperBlockingQueue::~MatWrapperBlockingQueue() {

}

}}}
