---
title: A comparison of Akka Persistence with Eventuate
layout: post
comments: true
author: "Martin Krasser"
header-img: "img/distributed.png"
---

<hr/>

**23.10.2015**: Several updates in all sections to cover recent questions and developments.

<hr/>

The following is an attempt to describe the similarities and differences between [Akka Persistence](http://doc.akka.io/docs/akka/2.4.0/scala/persistence.html) and [Eventuate](http://rbmhtechnology.github.io/eventuate/). Both are Akka-based [event-sourcing](http://martinfowler.com/eaaDev/EventSourcing.html) and [CQRS](http://martinfowler.com/bliki/CQRS.html) tools written in Scala, making different distributed system compromises. For an introduction to these tools, please take a look at their online documentation.

I’m the original author of both, Akka Persistence and Eventuate, currently focusing exclusively on the development of Eventuate. Of course, I’m totally biased ;) Seriously, if I have goofed something, please let me know.

Command side
------------

In Akka Persistence, the command side (C of CQRS) is represented by `PersistentActor`s (PAs), in Eventuate by `EventsourcedActor`s (EAs). Their internal state represents an application’s write model. 

PAs and EAs validate new commands against the write model and if validation succeeds, generate and persist one or more events which are then handled to update internal state. After a crash or a normal application re-start, internal state is recovered by replaying persisted events from the event log, optionally starting from a snapshot. PAs and EAs also support sending messages with at-least-once delivery semantics to other actors. Akka Persistence provides the `AtLeastOnceDelivery` trait for that purpose, Eventuate the `ConfirmedDelivery` trait.

From this perspective PAs and EAs are quite similar. A major difference is that PAs must be singletons whereas EAs can be replicated and updated concurrently. If an Akka Persistence application accidentally creates and updates two PA instances with the same `persistenceId`, the underlying event log will be corrupted, either by overwriting existing events or by appending conflicting events. Akka Persistence event logs are designed for having only a single writer and cannot be shared. 

In Eventuate, EAs can share an event log. Events emitted by one EA can be consumed by other EAs, based on predefined and customizable event routing rules. In other words, EAs can collaborate by exchanging events over a shared event log. This collaboration can be, for example, a distributed business process executed by EAs of different type, or state replication where EAs of the same type reconstruct and update internal state at multiple locations. These locations can even be globally distributed. Event replication between locations is asynchronous and reliable.

Event relations
---------------

In Akka Persistence, events have a total order per PA but events emitted by different PAs are not related. Even if an event emitted by one PA actually happened before an event emitted by another PA, this relationship is not tracked by Akka Persistence. For example, if PA<sub>1</sub> persists an event e<sub>1</sub>, then sends a command to PA<sub>2</sub> which in turn persists another event e<sub>2</sub> during handling of that command, e<sub>1</sub> obviously happened before e<sub>2</sub> but applications cannot determine this relationship by comparing e<sub>1</sub> with e<sub>2</sub>.

Eventuate additionally tracks the happened-before relationship (= potential causality) of events. For example, if EA<sub>1</sub> persists an event e<sub>1</sub> and EA<sub>2</sub> persists an event e<sub>2</sub> after having consumed e<sub>1</sub>, then e<sub>1</sub> happened before e<sub>2</sub> which is also tracked. Happened-before relationships are tracked with [vector clocks](http://rbmhtechnology.github.io/eventuate/architecture.html#vector-clocks) and applications can determine whether any two events have a happened-before relationship or are concurrent by comparing their vector timestamps.

Tracking the happened-before relationship of events is a prerequisite for running multiple replicas of an EA. An EA that consumes events from its replicas must be able to determine whether its last internal state update happened before or is concurrent to (and potentially in conflict with) the consumed events. 

If the last internal state update happened before a consumed event, that event can be handled as regular update. If it is concurrent to the consumed event, the event might be a conflicting event and must be handled accordingly. If an EA’s internal state is a [CRDT](http://en.wikipedia.org/wiki/Conflict-free_replicated_data_type), for example, the conflict can be resolved automatically (see also Eventuate’s [operation-based CRDTs](http://rbmhtechnology.github.io/eventuate/user-guide.html#operation-based-crdts)). If internal state is not a CRDT, Eventuate provides further means to [track](http://rbmhtechnology.github.io/eventuate/user-guide.html#tracking-conflicting-versions) and [resolve](http://rbmhtechnology.github.io/eventuate/user-guide.html#resolving-conflicting-versions) conflicts, either automatically or interactively.

Event logs
----------

As already mentioned, in Akka Persistence each PA has its own private event log. Depending on the storage backend, an event log is either stored redundantly on several nodes (e.g. synchronously replicated for stronger durability guarantees) or stored locally. In either case, Akka Persistence requires a strongly consistent view on an event log. 

For example, a PA that crashed and recovers on another node must be able to read all previously written events in correct order, otherwise recovery may be incomplete and the PA may later overwrite existing events or append new events to the log that are in conflict with existing but unread events. Therefore, only storage backends that support strong consistency can be used for Akka Persistence.

The write availability of an Akka Persistence application is constrained by the write availability of the underlying storage backend. According to the [CAP theorem](http://en.wikipedia.org/wiki/CAP_theorem), write availability of a strongly consistent, distributed storage backed is limited. Consequently, the command side of an Akka Persistence application chooses CP from CAP. 

These constraints make it difficult to globally distribute an Akka Persistence application as strong consistency and total event ordering also require global coordination. Eventuate goes one step further here: it requires strong consistency and total event ordering only within a so called *location*. A location can be a data center, a (micro-)service, a node in a cluster or a process on a single node, to mention a few examples.

An Eventuate application that only consists of a single location implements the same consistency model as an Akka Persistence application. However, Eventuate applications usually consist of multiple locations. Events generated at individual locations are asynchronously and reliably replicated to other locations. Inter-location event replication is Eventuate-specific and preserves causal event storage order. Storage backends at different locations do not communicate directly with each other. Therefore, different storage backends can be used at different locations.

An Eventuate [event log](http://rbmhtechnology.github.io/eventuate/architecture.html#event-logs) that is replicated across locations is called a *replicated event log*, its representation at a given location is called a *local event log*. EAs deployed at different locations can exchange events by sharing a replicated event log. This allows for EA state replication across locations. EAs and their underlying event logs remain writeable even during inter-location network partitions. From this perspective, a multi-location Eventuate application chooses AP from CAP. Writes during a network partition at different locations may cause conflicts which can be resolved as described previously.

By introducing partition-tolerant locations, a global total ordering of events is not possible any more. The strongest partial ordering that is possible under these constraints is causal ordering i.e. an ordering that preserves the happened-before relation of events. In Eventuate, every location guarantees the delivery of events in causal order to their local EAs (and views, see [next section](#query-side)). The delivery order of concurrent events may differ at individual locations but is repeatable within a given location.

Query side
----------

In Akka Persistence, the query side (Q of CQRS) can be implemented with `PersistentView`s (PVs). A PV is currently limited to consume events from only one PA. This limitation has been [intensively discussed](https://groups.google.com/forum/#!msg/akka-user/MNDc9cVG1To/blqgyC7sIRgJ) on the Akka mailing list. A proposed solution, available since Akka 2.4, is [Akka Persistence Query](http://doc.akka.io/docs/akka/2.4.0/scala/persistence-query.html): storage plugins may provide support for aggregating events from multiple PAs and serve the result as [Akka Streams](http://doc.akka.io/docs/akka-stream-and-http-experimental/1.0/scala.html) `Source`.

In Eventuate, the query side can be implemented with `EventsourcedView`s (EVs). An EV can consume events from all EAs that share an event log, even if they are globally distributed. Events are always consumed in correct causal order. An application can either have a single replicated event log or several event logs, organized around topics, for example. Future extensions will allow EVs to consume events from multiple event logs. An Akka Streams API in Eventuate is also planned.

Storage plugins
---------------

From a storage plugin perspective, events in Akka persistence are primarily organized around `persistenceId` i.e. around PA instances having their own private event log. Aggregating events from several PAs requires either the creation of an additional index in the storage backend or an on-the-fly event stream composition when serving a query. In Eventuate, events from several EAs are stored in the same shared event log. During recovery, EAs that don’t have an `aggregateId` defined, consume all events from the event log while those with a defined `aggregateId` only consume events with that `aggregateId` as routing destination. This requires Eventuate storage plugins to maintain a separate index from which events can be replayed by `aggregateId`.

Akka Persistence has a public storage plugin API for journals and snapshot stores with many implementations [contributed by the community](http://akka.io/community/). Eventuate will also define a public storage plugin API in the future. At the moment, applications can choose between a [LevelDB storage backend](http://rbmhtechnology.github.io/eventuate/reference/event-log.html#leveldb-storage-backend) and a [Cassandra storage backend](http://rbmhtechnology.github.io/eventuate/reference/event-log.html#cassandra-storage-backend).

Throughput
----------

Both, PAs in Akka Persistence and EAs in Eventuate can choose whether to keep internal state in sync with the event log. This is relevant for applications that need to validate new commands against internal state before persisting new events. To prevent validation against stale state, new commands must be delayed until a currently running event write operation successfully completed. PAs support this with a `persist` method (in contrast to `persistAsync`), EAs with a `stateSync` boolean property.

A consequence of synchronizing internal state with an event log is decreased throughput. Synchronizing internal state has a stronger impact in Akka Persistence than in Eventuate because of the details how event batch writes are implemented. In Akka Persistence, events are batched on PA level, but only when using `persistAsync`. In Eventuate there’s a separate batching layer between EAs and the storage plugin, so that events emitted by different EA instances, even if they sync their internal state with the event log, can be batched for writing. 

Comparing the write throughput of two single PA and EA instances, they are approximately the same in Akka Persistence and Eventuate (assuming a comparable storage plugin). However, in Eventuate, the overall write throughput can increase with an increasing number of EA instances, whereas the write throughput in Akka Persistence can not. This is especially relevant for applications that follow a one PA/EA per [aggregate](http://martinfowler.com/bliki/DDD_Aggregate.html) design with thousands to millions active (= writable) instances. Looking at the Akka Persistence code, I think it shouldn’t be too much effort moving the batching logic of PA down to a separate batching layer.

Conclusion
----------

Eventuate supports the same consistency model as Akka Persistence but additionally allows relaxation to causal consistency. This relaxation is necessary for EA high-availability and partition tolerance (AP of CAP). Eventuate also supports reliable actor collaboration based on causally ordered, de-duplicated event streams. From these perspectives, Eventuate is a functional superset of Akka Persistence. 

Choosing availability over consistency requires that conflict detection and resolution (either automated or interactive) must be a primary concern. Eventuate supports that by providing operation-based CRDTs as well as utilities and APIs for tracking and resolving conflicting versions of application state.

Handling conflicts rather than preventing them is important for the resilience of distributed systems. Being able to remain operable within a location that is temporarily partitioned from other locations also makes Eventuate an interesting option for offline use cases. 

Eventuate is still a young project. It started as a prototype in late 2014 and was [open-sourced](https://github.com/RBMHTechnology/eventuate) in 2015. It is actively developed in context of the [Red Bull Media House](http://www.redbullmediahouse.com/) (RBMH) [Open Source Initiative](http://rbmhtechnology.github.io/) and primarily driven by internal RBMH projects.
