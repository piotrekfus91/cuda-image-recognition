#ifndef BLOCKINGQUEUE_H_
#define BLOCKINGQUEUE_H_

#include <queue>
#include <boost/thread.hpp>

namespace cir { namespace devenv { namespace concurrency {

template<class T>
class BlockingQueue {
public:
	BlockingQueue(int size) : _size(size) {

	}

	virtual ~BlockingQueue() {

	}

	void add(T elem) {
		boost::mutex::scoped_lock lock(_mutex);
		while(_queue.size() >= _size) {
			_fullCond.wait(lock);
		}
		_queue.push(elem);
		lock.unlock();
		_emptyCond.notify_all();
	}

	T get() {
		boost::mutex::scoped_lock lock(_mutex);
		while(_queue.size() == 0) {
			_emptyCond.wait(lock);
		}
		T elem = _queue.front();
		_queue.pop();
		lock.unlock();
		_fullCond.notify_all();
		return elem;
	}

private:
	unsigned int _size;
	std::queue<T> _queue;
	boost::mutex _mutex;
	boost::condition_variable _emptyCond;
	boost::condition_variable _fullCond;
};

}}}
#endif /* BLOCKINGQUEUE_H_ */
