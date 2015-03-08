#include "cir/common/concurrency/IndexedMatWrapper.h"

namespace cir { namespace common { namespace concurrency {

IndexedMatWrapper::IndexedMatWrapper() {
	id = -1;
}

IndexedMatWrapper::~IndexedMatWrapper() {

}

bool IndexedMatWrapper::isPoison() {
	return id == -1;
}

}}}
