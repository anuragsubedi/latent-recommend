<primary_system_instructions>
You are an expert Data Engineer and Python Developer. We are building the foundational data pipeline for a content-based music recommendation engine. Your overarching mission is to build the complete data infrastructure for `latent-recommend`, a purely content-based music recommendation and similarity search engine.

Unlike traditional platforms that rely on collaborative filtering, this system will eventually map the acoustic topology of music using latent vectors extracted from diffusion transformer models.

...
...
...

</primary_system_instructions>

---

<context_documents>


Before proceeding, ingest the following context files to understand the overarching system architecture.

1. Initial Roadmap: Read the "initial_roadmap" documentation I wrote for a project that I'm about to start building for my "Statistical Machine Learning" course --> @beautifulMention.

2. Generative Model Context: @beautifulMention (Provides context on the Deep & Generative AI counterpart to this project).

3. ACE-Step Model Card: read this document  get the preliminary details of the latest open-source foundational model for music -->@beautifulMentionto

4. ACE-Step Ultimate Guide: @beautifulMention (WARNING: This is a massive document. Do not attempt to summarize or parse this entirely. Use semantic search to extract ONLY the necessary specifications regarding the VAE/Encoder bottleneck output dimensions, as we will need to store these latent vectors in our database later).

5. ACE-Step Technical Paper: @beautifulMention (Reference material for the generative diffusion architecture).


---

</context_documents>
   
---

<final_instructions>

With this preliminary information, assess and analyse the project and recall the initial roadmap  and think more about the Phase 1 to build the metadata skeleton and the preliminary skeleton. 

FYI, here's what I thought about the steps for the Phase 1:
- Generate an Artifact containing a detailed Implementation Plan and a Database Schema (include tables for Track, Artist, Album, GenreTags, and a specific column for PopularityIndex). Pause execution and wait for user approval of this schema.  
- Once approved, write the Python ingestion script using \`sqlite3\` and \`requests\`.   
- Implement robust API rate-limiting handling (exponential backoff/retries) to comply with MusicBrainz and Last.fm API guidelines.  
- Write a secondary Python validation script that queries the populated SQLite database and prints a statistical summary of the dataset's distribution by Genre and PopularityIndex to mathematically verify that the bias mitigation strategy was successful. 


And since you're working with me on this entire thing, feel free to keep notes of whatever you feel might be relevant exploring and feel free to expand upon it with my acknowledgement. As you analyze the requirements, initialize a `design_notes.md` file in the workspace to log your architectural thoughts, assumptions, and any potential API bottlenecks you foresee.


Lets get started - Think about this, and then ask me any questions if you need clarity about anything. Make a step-by-step plan and ACKNOWLEDGE WITH ME BEFORE MAKING ANY MAJOR CODE CHANGES.  

</final_instructions>

 