---
title: Implementing operation-based CRDTs in Scala 
layout: post
comments: true
---

In a [previous post](http://krasserm.github.io/2015/01/13/event-sourcing-at-global-scale/) I described how actor state can be globally replicated via event sourcing. Keeping replicas available for writes during a network partition requires a resolution of conflicting writes when the partition heals. In this context, [conflict-free replicated data types](http://en.wikipedia.org/wiki/Conflict-free_replicated_data_type) (CRDTs) have already been mentioned.

The [eventuate](https://github.com/RBMHTechnology/eventuate) project now provides implementations of [operation-based CRDTs](http://en.wikipedia.org/wiki/Conflict-free_replicated_data_type#Operation-based_CRDTs) (CmRDTs) as specified in the paper [A comprehensive study of Convergent and Commutative Replicated Data Types](http://hal.upmc.fr/docs/00/55/55/88/PDF/techreport.pdf) by Marc Shapiro et. al. Currently, the following CmRDTs are implemented (more coming soon):

- [Counter](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/Counter.scala) (specification 5)
- [MV-Register](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/MVRegister.scala) (op-based version of specification 10) 
- [OR-Set](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/ORSet.scala) (specification 15)

Basis for the implementation is a [replicated event log](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/README.md#event-log) and [event-sourced actors](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/README.md#event-sourced-actors). A replicated event log

- supports the reliable broadcast of update-operations needed by CmRDTs.  
- chooses AP from [CAP](http://en.wikipedia.org/wiki/CAP_theorem) i.e. applications can continue writing to a local replica during a network partition. 
- preserves causal ordering of events which satisfies all _downstream_ preconditions of the CmRDTs specified in the paper.

Event-sourced actors

- generate vector timestamps that are used by some CmRDTs to determine whether any two updates are concurrent or causally related.
- maintain CmRDT instances in-memory. Their state can be recovered by replaying update-operations (optionally starting from a snapshot).

CmRDT update specifications in the paper have a close relationship to the command and event handler of an event-sourced actor. A CmRDT update has two phases. The first phase is called _atSource_: 

> It takes its arguments from the operation invocation; it is not allowed to make side effects; it may compute results, returned to the caller, and/or prepare arguments for the second phase.

To _prepare arguments for the second phase_, update-operation events are generated and persisted in the command handler. The second update phase is called _downstream_. It executes 

> ... immediately at the source, and asynchronously, at all other replicas; it can not return results.

The _downstream_ phase executes by consuming update-operation events in the event handler. Not only does the local replica consume the events it generated and persisted but also all other replicas on the same replicated event log consume these events (= reliable broadcast). Consumed update-operation events finally change the state of CmRDTs that are managed by event-sourced actors.

Usage
-----

Applications that want to use these CmRDTs don't need to interact directly with event-sourced actors. Instead, eventuate provides service interfaces for reading and updating CmRDTs. There's a service interface for each supported CmRDT type.

### Counter

`Counter[A]` CmRDTs are managed by `CounterService[A]` which provides the following read and update operations. 

    class CounterService[A : Integral](val processId: String, val log: ActorRef) {
      def value(id: String): Future[A] = { ... }
      def update(id: String, delta: A): Future[A] = { ... }
    }

The `value` method reads a counter value, `update` updates a counter value with a given `delta`. Counter instances are identified by application-defined `id`s. Each `CounterService` replica must have a unique `processId` and a reference to the replicated event `log`. The following example creates and uses a `CounterService[Int]` for reading and updating `Counter[Int]` CmRDTs.

    import akka.actor.ActorRef
    import com.rbmhtechnology.eventuate.crdt.CounterService

    val eventLog: ActorRef = ...
    val counterService = new CounterService[Int]("counter-replica-1", eventLog)

    // counter-1 usage
    counterService.update("counter-1", 11) // increment
    counterService.update("counter-1", -2) // decrement
    counterService.value("counter-1")      // read

    // counter-2 usage
    counterService.value("counter-2")     // read
    counterService.update("counter-2", 3) // increment

Counters are created on demand if referenced the first time by an update operation.

### MV-Register

`MVRegister[A]` CmRDTs are managed by `MVRegisterService[A]` which provides the following read and update operations.

    class MVRegisterService[A](val processId: String, val log: ActorRef) {
      def value(id: String): Future[Set[A]] = { ... }
      def set(id: String, value: A): Future[Set[A]] = { ... }
    }

### OR-Set

`ORSet[A]` CmRDTs are managed by `ORSetService[A]` which provides the following read and update operations.

    class ORSetService[A](val processId: String, val log: ActorRef) {
      def value(id: String): Future[Set[A]] = { ... }
      def add(id: String, entry: A): Future[Set[A]] = { ... }
      def remove(id: String, entry: A): Future[Set[A]] = { ... }
    }

Running
-------

A running OR-Set example, based on a multi-JVM test, can be found [here](https://github.com/RBMHTechnology/eventuate/blob/blog-crdt/src/multi-jvm/scala/com/rbmhtechnology/eventuate/crdt/ReplicatedORSetSpec.scala).
