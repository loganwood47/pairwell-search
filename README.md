# PairWell

### UCLA Anderson MBA Capstone: AI-Powered Nonprofit Discovery & Matching Platform

**PairWell is a two-sided marketplace designed to connect corporate donors with nonprofits through personalized, push-based recommendations.**

This project was developed as the **capstone project for my MBA at UCLA Anderson**, combining customer discovery, market research, product strategy, business modeling, financial planning, and an AI/ML prototype into a single venture concept.

My primary contribution was the product and technical architecture including the recommendation engine, data/ML pipeline, prototype application, as well as the product strategy and business model.

**[Try the live prototype →](https://pairwell-search.streamlit.app/)**

---

## The Problem

Corporate giving is often a fragmented, search-heavy process. Organizations must identify nonprofits, evaluate their credibility and impact, obtain internal approval, and manage donations and reporting across multiple tools.

At the same time, small and mid-sized nonprofits struggle with visibility and donor acquisition.

PairWell's hypothesis was that the process could be improved by shifting from **pull-based search to proactive, personalized discovery**: instead of asking donors to search through thousands of organizations, PairWell recommends nonprofits and projects based on their interests, geography, and desired impact.

---

## The Product

PairWell was designed as a unified platform connecting two sides of the social-impact ecosystem:

**For donors**

* Personalized nonprofit recommendations
* Discovery of projects aligned with giving priorities
* Impact and progress tracking
* Nonprofit verification
* Centralized donation management

**For nonprofits**

* Profiles and project-level fundraising pages
* Increased discoverability
* Donor engagement and storytelling
* Impact reporting
* Access to new corporate funding opportunities

The initial product focused on monetary donations, with in-kind giving and volunteering identified as potential future extensions.

---

# From Business Idea to Working Prototype

The project deliberately combined **business strategy and technical implementation**.

### 1. Customer Discovery

We conducted interviews and surveys with both nonprofit and corporate-donor stakeholders to identify pain points, feature preferences, and willingness to pay.

Our research identified a particularly attractive early-adopter segment among mid-market companies with significant friction around nonprofit discovery, verification, approvals, and impact reporting. On the nonprofit side, the strongest initial fit was small-to-mid-sized organizations that needed better fundraising infrastructure but lacked the resources of large national nonprofits.

### 2. Market & Competitive Analysis

We analyzed the social-impact technology landscape across CSR platforms, fundraising tools, and volunteer platforms.

The central strategic hypothesis was that existing solutions were largely **siloed**, creating an opportunity to connect discovery, engagement, giving, and impact tracking in a single experience.

### 3. Marketplace & GTM Strategy

A two-sided marketplace creates an immediate cold-start problem: donors need nonprofits to make the platform useful, while nonprofits need donors.

Our solution was to initially prioritize the **supply side**, using nonprofit data to create a useful discovery experience before attempting to scale both sides simultaneously.

## As the platform grows, the model is designed to benefit from cross-side network effects and a data flywheel: more nonprofits create more value for donors, more donors create more value for nonprofits, and interactions between the two sides create additional data for personalization.

# Recommendation Engine

The technical centerpiece of the project is a hybrid recommendation system designed to match donors with nonprofits.

The initial scoring model combines four signals:

```text
Total Score =
    α × Mission Similarity
  + β × Geographic Similarity
  + γ × PairWell Score
  + δ × Learned Model Score
```

### Mission Similarity

A donor describes the causes they care about. An LLM expands those preferences into a representative "ideal nonprofit" mission, which is then embedded using a Sentence Transformer.

Nonprofit mission statements are embedded in the same vector space, allowing semantic similarity to be calculated between the donor and nonprofit.

### Geographic Similarity

Donor and nonprofit locations are geocoded and geographic distance is calculated using the Haversine formula, producing a normalized proximity score.

### Learned Similarity

The longer-term goal is for a two-tower recommendation model to learn donor–nonprofit affinity from behavioral interactions.

## Because real interaction data was initially unavailable, the prototype used synthetic users to develop and test the architecture. The strategy was to gradually shift weighting toward the learned model as authentic engagement data accumulated.

# Technical Architecture

```text
                     ┌──────────────────┐
                     │   Donor Profile  │
                     └────────┬─────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Interest Processing│
                    │      + LLM         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Mission Embedding │
                    └─────────┬──────────┘
                              │
                              ▼
┌────────────────┐    ┌──────────────────┐
│ Nonprofit Data │───▶│ Vector Similarity│
└────────────────┘    └────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Hybrid Recommendation│
                    │       Model         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Ranked Nonprofits  │
                    └────────────────────┘
```

**Stack:** Python · PyTorch · FAISS · Sentence Transformers · PostgreSQL · Streamlit

The repository includes the application code, tests, and CI configuration used to develop the prototype.

---

# Business Model

PairWell was designed as a subscription-based SaaS platform serving both sides of the marketplace.

The business plan modeled three subscription tiers, donation-processing revenue, and potential value-added services.

The financial model was built as a dynamic 60-month model incorporating user acquisition, conversion, churn, pricing, marketing channels, expenses, and cash flow. The base case projected approximately **$87K in Year 1 revenue and $5.2M+ in Year 5 revenue**, while explicitly identifying acquisition efficiency, conversion, churn, pricing sensitivity, and other assumptions that would need validation with real customers.

---

# What I Learned

The most valuable part of PairWell was was working through the interaction between **product, market, and technology constraints**, rather than the model itself

A few examples:

* A technically strong recommendation engine is useless without enough supply and demand to make recommendations valuable.
* A two-sided marketplace cannot simply launch both sides simultaneously; the cold-start problem needs to be designed around.
* Synthetic data can validate a technical architecture, but it cannot substitute for real behavioral data when evaluating a production recommendation system.
* The right ML objective ultimately depends on the business objective: recommendation quality should eventually be evaluated against real donor behavior and downstream outcomes, not simply model similarity.

---

# Project Status

PairWell remains a **prototype rather than a production product**.

The current implementation demonstrates the core recommendation experience and technical architecture. The next stage would be validating the business assumptions with real customers and replacing synthetic behavioral data with real donor–nonprofit interaction data.

---

## Links

**[Live Prototype →](https://pairwell-search.streamlit.app/)**

**[MBA Business Plan →](./PairWell_Business_Plan.pdf)**

---

## Built As

**MBA Capstone Project — UCLA Anderson School of Management**
