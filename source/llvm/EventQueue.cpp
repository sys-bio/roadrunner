/*
 * EventQueue.cpp
 *
 *  Created on: Aug 16, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EventQueue.h"
#include "LLVMExecutableModel.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include <algorithm>
#include <cstring>
#include <iomanip>

static void dump_array(std::ostream &os, int n, const double *p)
{
    if (p)
    {
        os << std::setiosflags(std::ios::floatfield) << std::setprecision(8);
        os << '[';
        for (int i = 0; i < n; ++i)
        {
            os << p[i];
            if (i < n - 1)
            {
                os << ", ";
            }
        }
        os << ']' ;
    }
    else
    {
        os << "NULL" ;
    }
}

using namespace rr;

namespace rrllvm
{

Event::Event(LLVMExecutableModel& model, uint id) :
        model(model),
        id(id),
        delay(model.getEventDelay(id)),
        assignTime(delay + model.getTime()),
        dataSize(model.getEventBufferSize(id)),
        data(new double[model.getEventBufferSize(id)])
{
    if (model.getEventUseValuesFromTriggerTime(id))
    {
        model.getEventData(id, data);
    }
    else
    {
        std::memset(data, 0, dataSize * sizeof(double));
    }

    Log(Logger::LOG_DEBUG) << "created event at time " << model.getTime() <<
            ": " << *this;
}

Event::Event(const Event& o) :
        model(o.model),
        id(o.id),
        delay(o.delay),
        assignTime(o.assignTime),
        dataSize(o.dataSize),
        data(new double[o.dataSize])
{
    std::copy(o.data, o.data + o.dataSize, data);
}

Event& Event::operator=(const Event& rhs )
{
    delete data;
    model = rhs.model;
    id = rhs.id;
    delay = rhs.delay;
    assignTime = rhs.assignTime;
    dataSize = rhs.dataSize;
    data = new double[rhs.dataSize];
    std::copy(rhs.data, rhs.data + rhs.dataSize, data);
    return *this;
}

Event::~Event()
{
    delete[] data;
}

bool Event::isExpired() const
{
    return !(model.getEventTrigger(id) || model.getEventPersistent(id));
}

bool Event::isCurrent() const
{
    return delay == 0.0 && (model.getEventPersistent(id) ||
            model.getEventTrigger(id));
}

double Event::getPriority() const
{
    return model.getEventPriority(id);
}

void Event::assign() const
{
    if (!model.getEventUseValuesFromTriggerTime(id))
    {
        model.getEventData(id, data);
    }
    Log(Logger::LOG_DEBUG) << "assigning event: " << *this;
    model.assignEvent(id, data);
}

bool Event::isPersistent() const
{
    return model.getEventPersistent(id);
}

bool Event::useValuesFromTriggerTime() const
{
    return model.getEventUseValuesFromTriggerTime(id);
}

bool Event::isTriggered() const
{
    return model.getEventTrigger(id);
}

bool operator<(const Event& a, const Event& b)
{
    if (a.assignTime != b.assignTime)
    {
        return a.assignTime < b.assignTime;
    }
    else
    {
        double ap = a.getPriority();
        double bp = b.getPriority();
        return ap > bp;
    }
}

bool Event::isRipe() const
{
    return ((isPersistent() || isTriggered()) &&
            (delay == 0.0 || assignTime <= model.getTime()));

}

std::ostream& operator <<(std::ostream& os, const Event& event)
{
    os << "Event{ " << event.id << ", " <<
            event.model.getEventTrigger(event.id) << ", " <<
            event.isExpired() << ", " << event.isCurrent() << ", " <<
            event.getPriority() << ", " << event.delay << ", " <<
            event.assignTime << ", ";

    dump_array(os, event.dataSize, event.data);

    os << "}";

    return os;
}

struct EventPredicate
{
    EventPredicate(uint eventId) : eventId(eventId) {};

    bool operator()(const rrllvm::Event& event) const
    {
        return event.id == eventId;
    }

    uint eventId;
};

bool EventQueue::eraseExpiredEvents()
{
    bool erased = false;
    iterator i = sequence.begin();
    while (i != sequence.end())
    {
        if (!(*i).isExpired())
        {
            ++i;
        }
        else
        {
            Log(Logger::LOG_DEBUG) << "removing expired event: " << *i;
            i = sequence.erase(i);
            erased = true;
        }
    }
    return erased;
}

bool EventQueue::hasCurrentEvents()
{
    return size() && top().isCurrent();
}



bool EventQueue::applyEvents()
{
    bool applied = false;
    if (sequence.size())
    {
        Log(Logger::LOG_DEBUG) << "event list before sort: " << *this;

        sequence.sort();

        Log(Logger::LOG_DEBUG) << "event list after sort, before apply: " << *this;

        std::deque<iterator> ripe;
        iterator i = sequence.begin();
        while (i != sequence.end())
        {
            if (i->isRipe())
            {
                ripe.push_back(i);
            }

            const Event& a = *i;
            ++i;
            if (i != sequence.end() && a < (*i))
            {
                break;
            }
        }

        Log(Logger::LOG_DEBUG) << "found " << ripe.size() << " ripe events";

        if (ripe.size())
        {
            uint index = std::rand() % ripe.size();
            iterator i = ripe[index];

            Log(Logger::LOG_DEBUG) << "assigning the " << index << "\'th item";
            (*i).assign();

            sequence.erase(i);

            applied = true;

            Log(Logger::LOG_DEBUG) << "event list after apply: " << *this;
        }
    }

    if(applied)
    {
        // erase doesn't care about heap property, but heapifies
        // if something was erased, we need to return the queue
        // in heapified state, so heapify in not already heapified
        // by erase.
        eraseExpiredEvents();
    }

    return applied;
}

uint EventQueue::size() const
{
    return sequence.size();
}

void EventQueue::push(const Event& e)
{
    sequence.push_back(e);
}

EventQueue::const_reference EventQueue::top()
{
    sequence.sort();
    return sequence.front();
}

double EventQueue::getNextPendingEventTime()
{
    if (size())
    {
        return top().assignTime;
    }
    else
    {
        return 0;
    }
}

void EventQueue::saveState(std::ostream& out)
{
	rr::saveBinary(out, sequence.size());
	for (Event e : sequence)
	{
		rr::saveBinary(out, e.id);
		rr::saveBinary(out, e.delay);
		rr::saveBinary(out, e.assignTime);
		rr::saveBinary(out, e.dataSize);
		out.write((char*)e.data, e.dataSize * sizeof(double));
	}
}

void EventQueue::loadState(std::istream& in, LLVMExecutableModel& model)
{
	size_t sequence_size;
	rr::loadBinary(in, sequence_size);
	for (size_t i = 0; i < sequence_size; i++)
	{
		uint id;
		rr::loadBinary(in, id);
		Event e(model, id);
		rr::loadBinary(in, e.delay);
		rr::loadBinary(in, e.assignTime);
		rr::loadBinary(in, e.dataSize);
		in.read((char*)e.data, e.dataSize * sizeof(double));
		push(e);
	}
}

std::ostream& operator<< (std::ostream& stream, const EventQueue& queue)
{
    stream << "EventQueue {" << std::endl;
    int j = 0;
    for(EventQueue::const_iterator i = queue.sequence.begin(); i != queue.sequence.end(); ++i)
    {
        stream << "event " << j++ << ": " << *i << std::endl;
    }
    stream << "}";

    return stream;
}

}




