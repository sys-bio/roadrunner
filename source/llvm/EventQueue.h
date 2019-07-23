/*
 * EventQueue.h
 *
 *  Created on: Aug 16, 2013
 *      Author: andy
 */

#ifndef EVENTQUEUE_H_
#define EVENTQUEUE_H_

#include "rrOSSpecifics.h"
#include <deque>
#include <queue>
#include <list>
#include <ostream>


namespace rrllvm {

class LLVMExecutableModel;

class Event
{
public:
    Event(LLVMExecutableModel&, uint id);
    Event(const Event& other);
    Event& operator=( const Event& rhs );
    ~Event();


    void assign() const;

    bool isExpired() const;

    /**
     * delay is zero and either persistent or triggered.
     */
    bool isCurrent() const;

    double getPriority() const;

    bool isPersistent() const;

    bool useValuesFromTriggerTime() const;

    bool isTriggered() const;

    /**
     * is this event ready to be applied
     */
    bool isRipe() const;


    LLVMExecutableModel& model;
    uint id;
    double delay;
    double assignTime;
    uint dataSize;

    /**
     * data block where assignment rules evaluations are stored
     * if useValuesFromTriggerTime is set.
     *
     * TODO: this should probably be allocated with the
     * ModelData structure. Original idea was that more than one
     * event of a type may be queued, but I don't think this is
     * possible.
     */
    double* data;


    friend bool operator<(const Event& a, const Event& b);

};

std::ostream& operator <<(std::ostream& os, const Event& data);


class EventQueue
{
public:
    typedef std::list<rrllvm::Event> _Sequence;
    typedef std::less<_Sequence::value_type> _Compare;
    typedef _Sequence::const_iterator const_iterator;
    typedef _Sequence::iterator iterator;
    typedef _Sequence::const_reference const_reference;

    /**
     * remove expired events from the queue.
     */
    bool eraseExpiredEvents();

    /**
     * are any events current (delay == 0 and triggered or persistant)
     */
    bool hasCurrentEvents();

    /**
     * assign all of the top most events with the same priority
     * and remove them from the queue.
     *
     * @returns true if any events were assigned, false otherwise.
     */
    bool applyEvents();

    /**
     * number of events in the queue
     */
    uint size() const;

    /**
     * event with lowest time to assignment and highest priority.
     */
    const_reference top();

    /**
     * insert a new event into the queue.
     *
     * The queue is re-sorted.
     */
    void push(const Event& e);

    /**
     * the time the next event is sceduled to be assigned.
     */
    double getNextPendingEventTime();

	/*
	* Save this EventQueue in binary format to out
	*/
	void saveState(std::ostream& out);

    /*
	* Load the events stored in in and add them to this queue and the executable model model
	*/
	void loadState(std::istream& in, LLVMExecutableModel& model);

    friend std::ostream& operator<< (std::ostream& stream, const EventQueue& queue);

private:

    /**
     * the data structure that holds the collection of events.
     *
     * Currently use a list, sortable and constant time insertion.
     */
    _Sequence  sequence;
    _Compare   comp;
};

std::ostream& operator<< (std::ostream& stream, const EventQueue& queue);




} /* namespace rrllvm */



#endif /* EVENTQUEUE_H_ */
