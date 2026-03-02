## **4\. Initial Agent Prompt Directive (Phase 1 Execution)**

*This prompt is designed to be pasted directly into an autonomous coding agent (e.g., Google Antigravity) to initiate Phase 1\.*

You are an expert Data Engineer and Python Developer. We are building the foundational data pipeline for a content-based music recommendation engine. 

Your objective is to build a SQLite database and a Python ingestion script that fetches track metadata from the MusicBrainz API and the Last.fm API. 

CRITICAL REQUIREMENT \- MITIGATING POPULARITY BIAS: The ingestion script must NOT solely pull popular songs. It must implement a stratified sampling strategy that actively seeks out obscure, indie, and low-playcount tracks across diverse genres (specifically ensuring coverage of ambient, dub-techno, and classical, alongside mainstream genres) to guarantee a statistically balanced dataset.

Please execute the following steps in order:  
1\. Generate an Artifact containing a detailed Implementation Plan and a Database Schema (include tables for Track, Artist, Album, GenreTags, and a specific column for PopularityIndex). Pause execution and wait for user approval of this schema.  
2\. Once approved, write the Python ingestion script using \`sqlite3\` and \`requests\`.   
3\. Implement robust API rate-limiting handling (exponential backoff/retries) to comply with MusicBrainz and Last.fm API guidelines.  
4\. Write a secondary Python validation script that queries the populated SQLite database and prints a statistical summary of the dataset's distribution by Genre and PopularityIndex to mathematically verify that the bias mitigation strategy was successful.  
5\. Create a \`requirements.txt\` file for all necessary dependencies.

Do not begin writing code until the Phase 1 Implementation Plan Artifact has been explicitly reviewed and approved.
