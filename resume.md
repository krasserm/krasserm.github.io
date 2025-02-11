---
layout: page
title: "Resume"
header-img: "img/distributed.png"
header-includes:
  - \hypersetup{colorlinks=false,
            allbordercolors={0 0 0},
            pdfborderstyle={/S/U/W 1}}
---

# Martin Krasser

Freelance AI engineer with strong background in machine learning, agentic systems, distributed systems, and system integration. Specialized in developing reliable AI systems and operating them at scale in production environments. Extensive industry experience in both technical and leadership roles. Active open source contributor.


- [Homepage](https://martin-krasser.com)
- [GitHub](https://github.com/krasserm)
- [LinkedIn](https://linkedin.com/in/krasserm)

<p></p>

## Industry experience

### Founder, Chief Agents Nanny

01.2025 - present, [Gradion AI](https://gradion.ai)

**Agentic application development and consulting services**  
We grow a team of AI agents collaborating with us to run our own business. We combine solid software engineering skills with latest research and development in artificial intelligence to make agents more reliable and capable. We support clients in automating their business processes with agentic AI. 


### Senior Director Machine Learning, Lead Machine Learning Engineer

08.2023 - 09.2024, [Canto](https://www.canto.com/), Freelance

**Scaling and extending AI Visual Search**  
Development of a cloud-native platform for scaling Canto's [AI Visual Search](https://www.canto.com/product/ai-visual-search/) solutions to thousands of customers. Development of hybrid search algorithms and research prototypes for inclusion of customer-specific asset metadata into AI Visual Search. 
Responsible for all AI search related research and development efforts at Canto.

### Director Machine Learning, Lead Machine Learning Engineer

04.2018 - 08.2023, [MerlinOne](https://merlinone.com/), Freelance

**Merlin Accelerated Intelligence (AI) Suite**  
Development of an AI search engine for semantic image, video, sound and document search in the MerlinOne digital asset management system. Supports facial recognition for identity-constrained searches and image aesthetics assessment for selecting images with highest perceived quality. Tuning of AI models and search indices on customer-specific data. 
Running in production at several customer sites, including the publicly accessible [AP Newsroom](https://newsroom.ap.org/) (see also [press release](https://www.ap.org/media-center/press-releases/2023/millions-of-ap-images-and-video-now-available-on-single-platform-with-ai-powered-search/)). Responsible for all ML research and development efforts at MerlinOne. The success of the Merlin AI Suite was a major factor in the acquisition of MerlinOne by Canto in 2023.

### Machine learning sabbatical

05.2017 - 04.2018, [https://martin-krasser.com](https://martin-krasser.com)

**Sabbatical year**  
Deep dive into mathematics, statistics, "traditional" machine learning and deep learning. Certifications from online courses. Publication of exercise and research work in articles and open source projects. Equal focus on scientific and engineering aspects. 


### Distributed Systems Engineer

09.2014 - 08.2017, [Red Bull Media House](http://www.redbullmediahouse.com/), Freelance

**Global distribution of a digital asset management system**  
Global distribution of the data management layer of RBMH's in-house DAM system for low-latency and partition-tolerant access to local datacenters. The developed inter-datacenter replication mechanism provides causal consistency guarantees and supports convergence of application state under concurrent updates via [operation-based CRDTs](https://rbmhtechnology.github.io/eventuate/architecture.html#operation-based-crdts). The generic part of the solution was open-sourced as [Eventuate](https://rbmhtechnology.github.io/eventuate/overview.html) toolkit. 
Production deployment to multiple datacenters world-wide. Responsible for conception, architecture, design and implementation of Eventuate and its in-house applications. Eventuate is an evolution of [Akka Persistence](https://doc.akka.io/docs/akka/current/persistence.html) which I developed in another project.

### Distributed Systems Engineer

10.2015 - 07.2017, [agido GmbH](http://www.agido.com/), Freelance

**Streaming platform for sports betting applications**  
Development of a streaming platform for calculating [real-time odds and risk models](https://www.agido.com/projekte) in sports betting applications. Bets on odds from up to hundred bookmakers, each with several thousand constantly changing odds, are used to calculate models for adjusting a bookmaker's odds such that their risk is minimized. Development of streaming data analytics extensions for Eventuate. 
Consultation on event sourcing best practices for all relevant use cases and responsible for their implementation with the [Eventuate](https://rbmhtechnology.github.io/eventuate/overview.html) toolkit.

### Software Engineer, Software Architect

10.2013 - 02.2014, [Lightbend](http://www.lightbend.com/), Freelance

**Akka Persistence: actor state persistence via event sourcing**  
Development of [Akka Persistence](https://doc.akka.io/docs/akka/current/persistence.html) which enables stateful [Akka](https://akka.io/) actors to persist their state via event sourcing. Events are written to append-only storage which allows for very high transaction rates and efficient replication. A stateful actor is recovered by replaying stored events to the actor, allowing it to rebuild its internal state. 
Implementation in [numerous](https://github.com/search?q=%22akka-persistence%22&type=repositories) commercial and open-source projects. Responsible for all phases of the project, from initial idea to production quality code. Akka Persistence is an evolution of [Eventsourced](https://github.com/eligosource/eventsourced), a predecessor that I developed in a prior project.

### Software Engineer, Software Architect

03.2012 - 10.2013, [Eligotech BV](http://www.eligotech.com/), Freelance

**Low-latency, high-throughput e-wallet management web service**  
Developmenent of the persistence layer of an e-wallet management web service for customers in the online gambling industry. Implementation of an event sourcing architecture for supporting low-latency and high-throughput transactions. The core components of the persistence layer were open-sourced as [Eventsourced](https://github.com/eligosource/eventsourced) library. 
Responsible for conception, design and implementation of Eventsourced and its integration into Eligotech products.

### Software Engineer, Machine Learning Engineer

04.2011 - present, [https://martin-krasser.com](https://martin-krasser.com), Freelance

**Freelance software development and consulting services**  
Research and development for industry [machine learning](https://krasserm.github.io/stories/#machine-learning) projects, mainly deep learning. Software development services with a focus on backend software and distributed systems. Open source contributions to [machine learning](https://krasserm.github.io/stories/#machine-learning), [agentic systems](https://krasserm.github.io/stories/#agentic-systems), [distributed systems](https://krasserm.github.io/stories/#distributed-systems), [event sourcing](https://krasserm.github.io/stories/#event-sourcing) and [system integration](https://krasserm.github.io/stories/#system-integration) projects. 


### Lead Software Architect, Senior Software Engineer

01.2005 - 01.2011, [InterComponentWare AG](https://icw-global.com/)

**E-Health integration platform based on HL7 and IHE standards**  
Development of the [Open eHealth Integration Platform](https://oehf.github.io/ipf-docs/) (IPF), a platform for integrating healthcare information systems. IPF was open sourced in 2008 after several years of in-house development and application in customer projects. IPF's programming model is a domain-specific language (DSL) for implementing [enterprise integration patterns](https://www.enterpriseintegrationpatterns.com/) in healthcare-specific integration solutions, based on [HL7](http://www.hl7.org/) and [IHE](https://www.ihe.net/) standards. 
Production deployments in many healthcare integration solution world-wide. IPF is still actively maintained today, by contributors from several healthcare integration providers. Responsible for conception, design and implementation of IPF and its application in customer projects. Founder of the open source project and lead developer until 2010.

### Senior Software Architect, Senior Software Engineer

09.2000 - 12.2004, [LION Bioscience AG](http://www.lionbioscience.com/)

**Distributed scientific computing solution for a drug discovery pipeline**  
Development of a distributed computing solution for integrating chemical analysis tools in a drug discovery pipeline, with fault-tolerant scheduling of tool executions and aggregation of analysis results, for unified experience across research locations in different countries. Implementation of high-performance hierarchical clustering algorithms that reduced analysis times by 1-2 orders of magnitude. 
Running for several years in production at [Bayer AG](https://www.bayer.com/). Responsible for architecture, design and implementation of the solution and its continuous improvement based on close collaboration with researchers.

### Software Developer, Research Assistant

09.1999 - 08.2000, [University of Salzburg](http://uni-salzburg.at/), CAME

**Ab-initio protein structure prediction**  
Development of algorithms and software for ab-initio protein structure prediction. Application of statistical mechanics for protein structure optimization and evaluation. 


### Software Developer

07.1998 - 01.1999, [Austrian Red Cross](http://www.roteskreuz.at/)

**Patient transportation management system**  
Development of a patient transportation management system in the IT department of the Red Cross in Graz, Austria. Work done during my civil service at the Red Cross. 



## Education

### Master of Science (Mag. rer. nat.) in Chemistry

1997, Karl-Franzens Universität Graz

### Various courses in computer science

1996, Karl-Franzens Universität Graz


## Additional information

- [Open source projects](https://krasserm.github.io/open-source/)
- [Stories](https://krasserm.github.io/stories/)
- [Articles](https://krasserm.github.io/articles/)
- [Talks](https://krasserm.github.io/talks/)
- [Courses](https://krasserm.github.io/courses/)
- [Technologies](https://krasserm.github.io/technologies/)
