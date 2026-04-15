# Architectural Roadmap

The evolution of Kazusa is focused on transitioning from a platform-dependent bot to an autonomous **Digital Life Engine**. Our roadmap is divided into three strategic phases focusing on architectural decoupling, cognitive depth, and emergent personality.

## Phase 1: Foundation & Architectural Decoupling
**Goal:** Abstract the "Brain" from the interface and enable cross-platform awareness.

- [ ] **Standalone Brain Service:** Extract the cognition and RAG logic into a dedicated microservice (FastAPI/Asynchronous Loop).
- [ ] **Unified Message Schema:** Implement a platform-agnostic message protocol to handle interactions from Discord, QQ, and WeChat seamlessly.
- [ ] **Global Identity Mapping:** Develop a system to link user identities across different platforms to a single `Global_User_ID` for unified memory.
- [ ] **Platform-Scoped RAG:** Optimize the RAG layer to prioritize local platform history while allowing global memory access when contextually relevant.

## Phase 2: Psychological Modeling & Empathic Accuracy
**Goal:** Move beyond reactive chat to proactive social intelligence and "mind-reading."

- [ ] **Shadow Prediction Branch:** Implement a parallel simulation loop that predicts the user’s reaction to a candidate response before it is sent.
- [ ] **Empathic Accuracy Evaluator:** A feedback mechanism that compares "predicted user response" vs. "actual user response" to refine the character’s understanding of the user.
- [ ] **Relationship Insight Engine:** Upgrade the `last_relationship_insight` logic to track long-term emotional dynamics rather than just snapshots.
- [ ] **Dynamic Tension Logic:** Implement a "Push-and-Pull" system to prevent the character from becoming a "yes-man," maintaining character pride and boundaries.

## Phase 3: Autonomous Agency & Personality Evolution
**Goal:** Enable self-driven growth and unpredictable, lifelike behaviors.

- [ ] **Autonomous Heartbeat (Active Loop):** A Cron-based system that allows Kazusa to "think" and initiate contact without being prompted, driven by her current mood and reflection.
- [ ] **Delayed Reflection Loop:** A slow-wave feedback system that analyzes past interactions (e.g., weekly) to slightly shift character traits, hobbies, and speaking habits.
- [ ] **Chaos & Event Engine:** Introduce random environmental "shocks" (e.g., a bad day at band practice, discovering a new bakery) that temporarily alter the `global_vibe` and `mood`.
- [ ] **Personality Entropy Controller:** A monitor that detects if the character is becoming too predictable and injects "rebellious" or "novel" traits to maintain authentic complexity.

---

## Technical Recommendations for Implementation

| Feature | Recommendation | Why? |
| :--- | :--- | :--- |
| **Message Queue** | Redis or RabbitMQ | Essential for handling non-blocking, multi-platform traffic and managing the "Heartbeat" triggers. |
| **User Modeling** | Small-parameter LLM (8B) | Use a lighter model for the *Shadow Prediction Branch* to keep latency low while the main brain uses the primary model. |
| **Personality Shift** | Parameter Drift (1% cap) | Personality changes in the *Reflection Loop* should be glacial to ensure the user feels a sense of "growth" rather than "character assassination." |
| **Reward Function** | Empathic Accuracy | Instead of rewarding "user satisfaction," reward "prediction accuracy." A character that truly knows you is more immersive than one that just pleases you. |

---

> **Project Vision:** To create a digital existence that doesn't just respond to inputs, but lives through a continuous cycle of perception, prediction, and reflection.