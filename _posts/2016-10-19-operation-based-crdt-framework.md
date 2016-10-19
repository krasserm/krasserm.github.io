---
title: A service framework for operation-based CRDTs
layout: post
comments: true
author: "Martin Krasser"
header-img: "img/distributed.png"
---

This article introduces a framework for developing operation-based CRDT services. It is part of the [Eventuate](https://github.com/RBMHTechnology/eventuate) project and supports the integration of operation-based CRDTs into Eventuate’s reliable causal broadcast infrastructure. It exposes CRDT instances through an asynchronous service API, manages their durability and recovery and allows for CRDT replication up to global scale.

After a brief introduction to operation-based CRDTs, their relation to Eventuate’s [event sourcing](http://rbmhtechnology.github.io/eventuate/architecture.html#event-sourcing) and [event collaboration](http://rbmhtechnology.github.io/eventuate/architecture.html#event-collaboration) features is discussed to get a better understanding of the inner workings of the framework. An example then demonstrates how to use the framework to implement a concrete CRDT service and how to run multiple replicas of that service. The rest of the article covers production deployment considerations and gives an overview of planned features.

Operation-based CRDTs
---------------------

Conflict-free replicated data types (CRDTs) are replicated data types that eventually converge to the same state under concurrent updates. A CRDT instance can be updated without requiring coordination with its replicas. This makes CRDTs highly available for writes. CRDTs can be classified into state-based CRDTs (CvRDTs or convergent replicated data types) and operation-based CRDTs (CmRDTs or commutative replicated data types). State-based CRDTs are designed to disseminate state among replicas whereas operation-based CRDTs are designed to disseminate operations. 

- CmRDT replicas are guaranteed to converge if operations are disseminated through a reliable causal broadcast (RCB) middleware and if they are designed to be commutative for concurrent operations. 

- CvRDTs don’t require special guarantees from the underlying messaging middleware but require increasing network bandwidth for state dissemination with increasing state size. They converge if their state *merge* function is defined as a join: a least upper bound on a [join-semilattice](https://en.wikipedia.org/wiki/Join-semilattice). CvRDTs are not further discussed in this article.

The execution of a CmRDT operation is done in two phases, *prepare* and *effect* [\[2\]][2] (also called *atSource* and *downstream* in [\[1\]][1]): *prepare* is executed on the local replica. It looks at the operation and (optionally) the current state and produces a message, representing the operation, that is then disseminated to all replicas. *effect* applies the disseminated operation at all replicas.

Relation to event sourcing
--------------------------

The two CmRDT update phases, *prepare* and *effect*, are closely related to the update phases of event-sourced entities, *command handling* and *event handling*, respectively:

- During *command handling*, an incoming command is (optionally) validated against current state of the entity and, if validation succeeds, an event representing the effect of the command is written to the event log. This corresponds to producing an operation representation during the CmRDT *prepare* phase.

- During *event handling*, the written event is consumed from the event log and used to update the current state of the entity. This corresponds to applying the produced operation representation to CmRDT state locally during the effect *phase*.

Event sourcing toolkits usually distinguish these two phases and provide event-sourced entity abstractions that let application define custom command and event handlers. For example, [Akka Persistence](http://doc.akka.io/docs/akka/2.4/scala/persistence.html) provides [PersistentActor](http://doc.akka.io/api/akka/2.4/#akka.persistence.PersistentActor), Eventuate provides [EventsourcedActor](http://rbmhtechnology.github.io/eventuate/latest/api/index.html#com.rbmhtechnology.eventuate.EventsourcedActor). 

These are quite similar from a conceptual and API perspective but have one important difference: A `PersistentActor` can only consume self-emitted events whereas an `EventsourcedActor` can also consume events emitted by other event-sourced actors. And exactly this is needed to implement CmRDTs with event-sourced entities. They must be able to [collaborate](http://rbmhtechnology.github.io/eventuate/architecture.html#event-collaboration) so that multiple replicas of the same entity can be deployed.

CmRDTs in Eventuate are plain Scala objects that are [internally managed inside event-sourced CRDT actors](http://rbmhtechnology.github.io/eventuate/architecture.html#operation-based-crdts). CmRDTs encapsulate state and implement *prepare* and *effect* logic. Their concrete implementation follows the specifications in [\[1\]][1]. CRDT actors handle CmRDT update commands by persisting update operations to the event log in the *prepare* phase and delegate the execution of these operations to the CmRDT in the *effect* phase. An operation emitted by one event-sourced actor is not only consumed by that actor itself but also by all other event-sourced actors with the same `aggregateId`. This ensures that all replicas of a given CmRDT instance receive the update operation during the *effect* phase.

Reliable causal broadcast
-------------------------

Most CmRDTs specified in [\[1\]][1] require causal delivery order for update operations. Causal delivery is trivial to achieve with an event log that provides total order. But the availability of such an event log is limited because all updates to it must be coordinated if the event log itself is replicated (like a topic partition in a Kafka cluster, for example) or because it is not replicated at all. Consequently, the availability of CmRDTs that share a totally ordered event log is constrained by the availability of the underlying event log, which is not what we want.

What we want is a distribution of CmRDT replicas across *locations* (or availability zones) where each location has its own local event log that remains available for writes even if partitioned from other locations. Events written at one location are asynchronously and reliably replicated to other locations. The strongest global ordering that can be achieved in such a *replication network* of local event logs is *causal ordering* which is sufficient for CmRDTs to work as specified. 

A replication network of local event logs in Eventuate is called a [replicated event log](http://rbmhtechnology.github.io/eventuate/architecture.html#event-logs). Causality in a replicated event log is tracked with [vector clocks](http://rbmhtechnology.github.io/eventuate/architecture.html#vector-clocks) as *potential causality* (a partial order). The total order of events in a local event log at each location is consistent with potential causality. Events produced within a replication network can therefore be consumed in correct causal order at any location of that network. This property makes replicated event logs a reliable causal broadcast (RCB) mechanism as required by CmRDTs.

Inter-location event replication is Eventuate-specific and ensures that local storage order is consistent with potential causality. The pluggable [storage backend](http://rbmhtechnology.github.io/eventuate/architecture.html#storage-backends) of a local event log may additionally provide redundant storage of events if stronger durability is needed. Event replication within a storage backend is completely independent from inter-location event replication. Inter-location event replication also works between locations with different storage backends. 

CRDT service framework
----------------------

### CRDT service definition

The previous sections outlined how CmRDTs can be integrated into Eventuate’s event sourcing and event collaboration infrastructure. In order to free CmRDT developers and users from working with actor APIs directly, Eventuate provides a CmRDT service development framework that hides these low-level details. This framework is also the implementation basis for all CmRDTs that are part of Eventuate. Usage of the framework is demonstrated in the following by implementing an *MV-Register* CRDT service (full source code [here](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/MVRegister.scala)).

An *MV-Register* or *Multi-Value Register* CRDT is a memory cell that supports *assign* to update the cell value and *value* to query it. In case of concurrent updates, multiple values are retained (in contrast to a *Last-Writer-Wins Register*, for example, where one of the values takes precedence). In case of multiple values, applications can later reduce them to a single value. 

{% highlight scala %}
import com.rbmhtechnology.eventuate.VectorTime
import com.rbmhtechnology.eventuate.Versioned

case class MVRegister[A](versioned: Set[Versioned[A]] = Set.empty[Versioned[A]]) {
  def value: Set[A] = versioned.map(_.value)

  def assign(v: A, vectorTimestamp: VectorTime): MVRegister[A] = {
    // <-> operator returns true for concurrent vector timestamps
    
    val concurrent = versioned.filter(_.vectorTimestamp <-> vectorTimestamp)
    copy(concurrent + Versioned(v, vectorTimestamp))
  }
}
{% endhighlight %}

The `MVRegister[A]` class internally stores assigned values of type `A` together with the `vectorTimestamp` of the corresponding operation as `Versioned[A](value: A, vectorTimestamp: VectorTime)`. Here, `Set[Versioned[A]]` is used because multiple concurrent assignments are allowed. Vector timestamps are generated by Eventuate for each persisted event (i.e. operation) and can be used by applications to determine whether any two events are causally related or concurrent. The `assign` method retains all assignments that are concurrent to the new assignment. All other existing assignments can be assumed to causally precede the new assignment because of causal delivery order. The new assignment is then added to retained concurrent assignments. The current value of an `MVRegister[A]` can be obtained with `value`.

To provide a service API for `MVRegister[A]`, we define a `MVRegisterService[A]` class that extends `CRDTService[MVRegister[A], Set[A]]`. [CRDTService](http://rbmhtechnology.github.io/eventuate/latest/api/index.html#com.rbmhtechnology.eventuate.crdt.CRDTService) is an abstract service that manages CmRDT instances inside event-sourced actors and provides an asynchronous API for reading and updating those instances. The service-internal actors interact with the local event log for reading and writing CmRDT operations. The only concrete service method we need to implement is the asynchronous `assign` method that defines which operation to use for the `MVRegister[A]` update.

{% highlight scala %}
import akka.actor.ActorRef
import akka.actor.ActorSystem
import com.rbmhtechnology.eventuate.crdt.CRDTService
import com.rbmhtechnology.eventuate.crdt.CRDTServiceOps
import scala.concurrent.Future

class MVRegisterService[A]
    (val serviceId: String, val log: ActorRef)
    (implicit val system: ActorSystem, val ops: CRDTServiceOps[MVRegister[A], Set[A]])
  extends CRDTService[MVRegister[A], Set[A]] {

  def assign(id: String, value: A): Future[Set[A]] = op(id, AssignOp(value))
}

case class AssignOp(value: Any)
{% endhighlight %}

The `assign` service method creates an `AssignOp` instance, representing a persistent *assign* operation, that is first used by the local replica during the *prepare* phase, then written to the event log and finally used by the local and all remote replicas during the *effect* phase. `CRDTService[MVRegister[A], Set[A]]` also provides a `value(id: String): Future[Set[A]]` method for obtaining the current value of a local `MVRegister[A]` replica (usage shown in next section). 

For being able to work, `MVRegisterService[A]` requires an implicit `CRDTServiceOps[MVRegister[A], Set[A]]` instance in scope. [CRDTServiceOps](http://rbmhtechnology.github.io/eventuate/latest/api/index.html#com.rbmhtechnology.eventuate.crdt.CRDTServiceOps) is a type class that maps the CmRDT update phases *prepare* and *effect* to methods on CmRDT instances. It also defines how to obtain the current value from a CmRDT instance.

{% highlight scala %}
import com.rbmhtechnology.eventuate.DurableEvent

object MVRegister {
  implicit def MVRegisterServiceOps[A] = new CRDTServiceOps[MVRegister[A], Set[A]] {
    override def value(crdt: MVRegister[A]): Set[A] =
      crdt.value

    override def prepare(crdt: MVRegister[A], operation: Any): Option[Any] = 
      super.prepare(crdt, operation)

    override def effect(crdt: MVRegister[A], operation: Any, event: DurableEvent): MVRegister[A] = 
      operation match {
        case AssignOp(value) => crdt.assign(value.asInstanceOf[A], event.vectorTimestamp)
      }
  }
}
{% endhighlight %}

For `MVRegister[A]`, the `prepare` phase is a no-op and the default implementation from `super` should be used (or better, inherited). The `effect` phase uses the persistent `AssignOp` to call the `assign` method on the `MVRegister[A]` instance. The vector timestamp is taken from the `event` parameter that provides all event metadata. 

The [MVRegister](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/MVRegister.scala) CmRDT and its service are defined in the [eventuate-crdt](http://rbmhtechnology.github.io/eventuate/download.html#id2) module. Other CmRDTs provided by Eventuate are [Counter](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/Counter.scala), [LWW-Register](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/LWWRegister.scala), [OR-Set](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/ORSet.scala) and [OR-Cart](https://github.com/RBMHTechnology/eventuate/blob/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/ORCart.scala) (an OR-Set based shopping cart CmRDT).

### CRDT service usage

The following example (full source code [here](https://github.com/krasserm/eventuate-crdt-example/blob/master/src/main/scala/example/crdt/MVRegisterExample.scala)) sets up three locations, each running a local `MVRegisterService[String]` instance using the local event log for persisting *assign* operations. The local event logs are connected to each other via their `ReplicationEndpoint`s to form a replicated event log. This implements the reliable causal broadcast needed to disseminate *assign* operations to all replicas. The locations run all in the same JVM, each having its own `ActorSystem` (usually, locations run on different nodes in the same or even different data centers). 

{% highlight scala %}
import akka.actor.ActorSystem
import com.rbmhtechnology.eventuate.ReplicationConnection
import com.rbmhtechnology.eventuate.ReplicationEndpoint
import com.rbmhtechnology.eventuate.crdt.MVRegisterService
import com.rbmhtechnology.eventuate.log.leveldb.LeveldbEventLog
import com.typesafe.config.ConfigFactory

def config(port: Int): String =
  // set port in akka-remote config (omitted)


def service(locationId: String, port: Int, connectToPorts: Set[Int]): MVRegisterService[String] = {
  implicit val system: ActorSystem =
    ActorSystem(ReplicationConnection.DefaultRemoteSystemName, ConfigFactory.parseString(config(port)))

  val logName = "L"

  val endpoint = new ReplicationEndpoint(id = locationId, logNames = Set(logName),
    logFactory = logId => LeveldbEventLog.props(logId),
    connections = connectToPorts.map(ReplicationConnection("127.0.0.1", _)))

  endpoint.activate()

  new MVRegisterService[String](s"service-$locationId", endpoint.logs(logName))
}

val serviceA = service("A", 2552, Set(2553, 2554)) // at location A
val serviceB = service("B", 2553, Set(2552, 2554)) // at location B
val serviceC = service("C", 2554, Set(2552, 2553)) // at location C

// ...
{% endhighlight %}

We will use a single `MVRegister[String]` instance, identified by `crdtId`, with a replica at each location. When calling `assign` or `value` for the first time after service creation, the service tries to recover the local replica from the event log by replaying operations or initializes a new one if there are no persistent operations for that instance id.

{% highlight scala %}
val crdtId = "1"

serviceA.assign(crdtId, "abc").onSuccess {
  case r => println(s"assign result to replica at location A: $r")
}

serviceB.assign(crdtId, "xyz").onSuccess {
  case r => println(s"assign result to replica at location B: $r")
}

// wait a bit ...

Thread.sleep(1000)

serviceC.value(crdtId).onSuccess {
  case r => println(s"read result from replica at location C: $r")
}
{% endhighlight %}

The updates start when `serviceA` assigns its local replica the value `abc` and `serviceB` its local replica the value `xyz`. These operations are expected to be concurrent because `serviceB` will likely have finished the local update before having received the *assign* operation from `serviceA`. However, there is a small chance that this is not the case. We can find that out by reading the value from `serviceC`, after having given operation dissemination enough time. In case of concurrent operations, output similar to the following is generated:

    assign result to replica at location A: Set(abc)
    assign result to replica at location B: Set(xyz)
    read result from replica at location C: Set(abc, xyz)

If the read value from `serviceC` contains both assignments, the operations were concurrent. If it only contains a single assignment, the operations are causally related.

Production deployment considerations
------------------------------------

All operations executed on a single CRDT service instance go to the same event log which puts an upper limit on write throughput. Applications that want to scale writes should consider creating [multiple replicated event logs](http://rbmhtechnology.github.io/eventuate/reference/event-log.html#replicated-event-log) and partition CRDT service instances accordingly. On the other hand, applications with moderate write rates may also share the same event log among multiple CRDT services of different type. Eventuate properly isolates these services.

As long as the total number of updates per CRDT instance is moderate e.g. less than a few thousand updates it is not necessary to create snapshots of CRDT state. Eventuate event logs are indexed on CRDT id (i.e. `aggregateId`) so that operation replay per instance is reasonably fast. For a much larger number of updates per instance it is recommended to save snapshots. CRDT services provide a generic `save(id: String): Future[SnapshotMetadata]` method for saving CRDT state snapshots where `id` is the CRDT instance id.

All CRDT services provided by Eventuate serialize CRDT operations and snapshots with Google [protocol buffers](https://developers.google.com/protocol-buffers/). The corresponding serializers are configured as described in [Custom serialization](http://rbmhtechnology.github.io/eventuate/reference/event-sourcing.html#custom-serialization). When using the CRDT framework for the development of custom CRDT services, it is recommended to provide custom serializers as well, otherwise Akka’s default serializer i.e. Java serializer is used for serializing operations and snapshots.

Eventuate’s CRDTs and its reliable causal broadcast infrastructure have been heavily [tested under chaotic conditions](https://github.com/RBMHTechnology/eventuate-chaos/) with all currently available [storage backends](http://rbmhtechnology.github.io/eventuate/architecture.html#storage-backends). Chaos was generated by randomly injecting network partitions between Eventuate locations, within Cassandra clusters and between Eventuate locations and Cassandra cluster nodes. Furthermore, network packets have been dropped randomly too. 

In all tests, replicated state successfully converged at all locations, even after long-running chaos tests (approx. 1 hour). A single missing event or a duplicate delivered to in-memory CRDT replicas would have broken convergence and let the chaos tests fail. The failure handling consequences learned during the chaos tests are summarized in the [Failure handling](http://rbmhtechnology.github.io/eventuate/reference/event-sourcing.html#failure-handling) section of the Eventuate documentation. 

Planned features
----------------

In addition to the basic CRDTs that are currently part of Eventuate, we plan to work on tree and graph CRDTs as well as CRDTs for collaborative text editing. We are also working on a Java API for the framework. A Java API for the existing CRDT services [is already available](https://github.com/RBMHTechnology/eventuate/tree/master/eventuate-crdt/src/main/scala/com/rbmhtechnology/eventuate/crdt/japi).

Eventuate’s CRDTs currently follow the specifications in [\[1\]][1]. As explained in [\[2\]][2] these specifications can be improved to make them *pure* operation-based. A prerequisite for such an implementation is a *tagged* reliable causal broadcast (TRCB) which is a RCB middleware that additionally exposes causality metadata to the application. Eventuate already delivers vector timestamps together with events to applications and can therefore be used as TRCB. We currently think about factoring out the TRCB part of Eventuate into a separate open source project.

Switching to pure operation-based CRDTs also increases write throughput as explained in [ticket 301](https://github.com/RBMHTechnology/eventuate/issues/301). We also work on increasing replication throughput by parallelizing stages in the replication pipelines and reducing the replication protocol overhead.

References
----------

\[1\] [A comprehensive study of Convergent and Commutative Replicated Data Types][1], 2011, Marc Shapiro et. al.  
\[2\] [Making Operation-based CRDTs Operation-based][2], 2014, Carlos Baquero et. al.

[1]: http://hal.upmc.fr/file/index/docid/555588/filename/techreport.pdf
[2]: http://haslab.uminho.pt/sites/default/files/ashoker/files/opbaseddais14.pdf
