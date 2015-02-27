#include "cir/devenv/ThreadInfo.h"
#include <boost/thread.hpp>

namespace cir { namespace devenv {

int ThreadInfo::getNumberOfThreads() {
	return boost::thread::hardware_concurrency();
}

}}
