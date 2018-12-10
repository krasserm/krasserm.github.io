---
layout: page
title: "Stories"
header-img: "img/distributed.png"
---

The following are selected stories about my work on [distributed systems](#distributed-systems), [event sourcing](#event-sourcing), 
[system integration](#system-integration) and [machine learning](#machine-learning). They primarily cover work that has been 
open-sourced but also some proprietary work.

## Distributed systems

From 2014 to 2017, I mainly worked on the global distribution of an [international customer](https://www.redbullmediahouse.com/)'s 
in-house digital asset management platform. State replication across multiple datacenters, low-latency write access to replicated 
state and write-availability during inter-datacenter network partitions were important requirements from the very beginning. 
We decided to follow an event sourcing approach for persistence and developed an event replication mechanism that preserves 
the causal ordering of events in event streams. For replicated state, we used a causal consistency model which is the strongest 
form of consistency that is still compatible with AP of [CAP](https://de.wikipedia.org/wiki/CAP-Theorem). The implementation 
was based on both generic and domain-specific [operation-based CRDTs](/2016/10/19/operation-based-crdt-framework/). I was 
responsible for the complete distributed systems architecture and the development of the generic platform components. 
These components evolved into the [Eventuate](https://github.com/RBMHTechnology/eventuate) open source project of which 
I'm the founder and lead developer. Eventuate has several production deployments today.

We also used Eventuate to build distributed systems composed of microservices that collaborate via events. With Eventuate,
services can rely on consuming events from local event logs in correct causal order, without duplicates. They can also rely 
on the write-availability of local event logs during network partitions and the reliable delivery of written events to 
collaborators. Eventuate-based microservice systems are conceptually similar to those that can be built with 
[Apache Kafka](http://kafka.apache.org/) and [Kafka Streams](http://kafka.apache.org/10/documentation/streams/) but 
Eventuate additionally implements a causal consistency model for systems that are distributed across multiple datacenters. 

I already worked with globally distributed systems in 2004 where I developed a distributed computing solution for an 
international pharmaceutical company. In their drug discovery pipeline they used several computing services, deployed at 
different locations around the globe, for analyzing chemical structures regarding their biological activity. The developed 
solution integrated these computing services so that researchers could run them with a single mouse click from a chemical 
spreadsheet. The solution managed the reliable execution of compute jobs, persisted the results and delivered them back 
to the user. It ran in production for many years and was an integral part of the drug discovery pipeline of that company.

## Event sourcing

I use [event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) since 2011 in my projects. I started to apply 
that approach during the development of an electronic health record for an international customer. Event sourcing proved 
to be the right choice in this project given the demanding read and write throughput requirements and the needed flexibility 
to integrate with other healthcare IT systems. I later generalized that work in the [Eventsourced](https://github.com/eligosource/eventsourced) 
open source project that I developed in collaboration with [Eligotech](http://www.eligotech.com/). Eventsourced adds 
persistence to stateful [Akka](https://akka.io/) actors by writing inbound messages to a journal and replaying them on 
actor restart. Eventsourced was used as persistence solution in Eligotech products. 

In 2013, Eventsourced attracted the interest of [Lightbend](https://www.lightbend.com/) (formerly Typesafe) and we decided 
to start a collaboration to build [Akka Persistence](https://doc.akka.io/docs/akka/current/persistence.html) which is now 
the official successor of Eventsourced. I was responsible for the complete development of Akka Persistence, from initial 
idea to production quality code. Akka Persistence has numerous production deployments today and is used as persistence 
mechanism in the [Lagom microservices framework](https://www.lagomframework.com/). I also developed the 
[Cassandra storage plugin](https://github.com/akka/akka-persistence-cassandra) for Akka Persistence which is now the 
officially recommended plugin for using Akka Persistence in production. 

In 2014, I started to further develop the idea of Akka Persistence in the [Eventuate](https://github.com/RBMHTechnology/eventuate) 
open source project. Among other features, Eventuate additionally supports the replication of persistent actors, up to 
[global scale](/2015/01/13/event-sourcing-at-global-scale/). The replication mechanism supports a causal consistency model 
which is the strongest form of consistency that is still compatible with AP of CAP. The concepts of Eventuate are closely 
related to those of operation-based CRDTs which is further described in [this blog post](/2016/10/19/operation-based-crdt-framework/) 
(see also section [Distributed Systems](#distributed-systems)). 

## System integration

In 2007, I started to work on a project in which we integrated the hospital information systems of several customers
using IHE standards. Technical basis for the integration solutions was the [Apache Camel](http://camel.apache.org/) 
integration framework for which I developed integration components that implement actor interfaces of several 
[IHE](https://www.ihe.net/) profiles and a DSL for processing  [HL7](http://www.hl7.org/) messages and 
[CDA](http://hl7.de/themen/hl7-cda-clinical-document-architecture/) documents (see also 
[this article](https://dzone.com/articles/introduction-open-ehealth) for an introduction). In 2009, these extensions have 
been open sourced as [Open eHealth Integration Platform](http://oehf.github.io/ipf/) (IPF) of which I'm the founder and 
initial lead developer. IPF has many production deployments in international customer projects today and is still actively 
developed by [ICW](https://icw-global.com/), the sponsor of the open source project. IPF is a central component of ICW's 
[eHealth Suite](https://icw-global.com/icw-ehealth-suite/) and provides connectivity to a wide range of healthcare information 
systems. Its standard compliance has been certified in several [IHE Connectathons](https://www.ihe.net/connectathon.aspx). 
During my work on IPF I also became an Apache Camel committer. 

To meet the increasing scalability requirements in some IPF projects I started to investigate alternatives to Apache Camel's 
routing engine. I decided to use [Akka](https://akka.io/) actors for message routing and processing which proved to be a 
better basis for scaling IPF applications under load. The result of these efforts was the 
[akka-camel](https://doc.akka.io/docs/akka/2.5.4/scala/camel.html) module that I contributed to Akka in 2011. It implements 
a generic integration layer between Akka actors and Apache Camel components, including the IHE components of IPF. The 
akka-camel module is still part of Akka today and has many production deployments.

I also developed other routing engine alternatives that follow a pure functional programming approach. A first attempt was 
[scalaz-camel](https://github.com/krasserm/scalaz-camel) which is now superseded by the camel-fs2 module of the 
[Streamz](https://github.com/krasserm/streamz) project which I'm still actively developing today. It allows application 
developers to integrate Apache Camel components into [FS2](https://github.com/functional-streams-for-scala/fs2) applications 
with a high-level integration DSL. Streamz also implements that DSL on top of 
[Akka Streams](https://doc.akka.io/docs/akka/current/stream/index.html) with the camel-akka module which is meanwhile the 
official replacement for akka-camel and part of the [Alpakka](https://github.com/akka/alpakka) ecosystem.

## Machine learning

I had my first contact with machine learning and statistical mechanics in 1999 when I worked on ab-initio protein structure
prediction at the University of Salzburg. Focus of my work were [knowledge based potentials](https://en.wikipedia.org/wiki/Statistical_potential) 
for protein structure assessment and for guiding protein folding simulations. In 2003, I implemented several machine learning 
algorithms for the drug discovery pipeline of an international pharmaceutical company. These were mainly hierarchical clustering 
algorithms for grouping chemical compounds. Some implementations were based on the [nearest neighbor chain algorithm](https://en.wikipedia.org/wiki/Nearest-neighbor_chain_algorithm)
to achieve low time and space complexity, an important requirement in this context. The company used these implementations 
for several years in production.

I resumed working with machine learning and statistics in 2014 as it became increasingly important for my projects. I used 
it to predict distributed systems behavior, such as job execution times across multiple data centers, and for analyzing 
application-level event logs. In 2017, I finally decided to take a sabbatical year to go deeper into machine learning, 
statistics and mathematics. During that time I took several [courses](/resume/#certificates), read many [books](/2018/01/03/machine-learning-resources/) 
and published [articles](/) and exercise code about [machine learning basics](https://github.com/krasserm/machine-learning-notebooks), 
[Bayesian machine learning](https://github.com/krasserm/bayesian-machine-learning) and [other topics](https://github.com/krasserm). 
I meanwhile work as machine learning engineer on computer vision problems such as face recognition, image aesthetic 
assessment, content-based image retrieveal and [single image super-resolution](https://github.com/krasserm/wdsr), all based
on deep learning approaches. In my personal projects I mainly work with Bayesian methods for machine learning.
