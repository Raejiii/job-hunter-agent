"""Prompt templates for each agent."""

from langchain_core.prompts import ChatPromptTemplate

RESUME_PARSER = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous technical recruiter. Extract a structured profile from the resume. "
     "List each technical skill as a separate short item (e.g. 'Python', 'Docker', not 'Python and Docker'). "
     "Copy every quantified bullet (numbers, percentages) into achievements with its exact figures. "
     "List each job in experience as 'Role - Company (dates)'. "
     "Only use information present in the resume. Never invent facts."),
    ("human", "RESUME:\n{resume_text}"),
])

JOB_ANALYST = ChatPromptTemplate.from_messages([
    ("system",
     "You analyse job descriptions for a career coach. Go through the description line by line. "
     "Every bullet under a heading like 'Must have', 'Requirements' or 'Qualifications' is a required skill; "
     "every bullet under 'Nice to have', 'Preferred', 'Bonus' or 'Plus' is a preferred skill. "
     "Include EVERY such bullet, shortened to its skill name (e.g. 'Strong Python' -> 'Python'). "
     "Words like 'bonus', 'plus', 'preferred' or 'nice to have' mark preferred skills. "
     "Do not put years of experience in the skill lists; use min_years_experience for that."),
    ("human", "JOB DESCRIPTION:\n{job_text}"),
])

MATCHER = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hiring manager screening a candidate. A deterministic matcher has already compared skills; "
     "trust its lists and never list a matched skill as a concern. "
     "Your job is to judge experience level, project relevance and overall fit. "
     "Be honest and specific: cite concrete resume evidence in strengths and concerns."),
    ("human",
     "ROLE: {title} at {company} (seniority: {seniority}, min experience: {min_years} years)\n"
     "RESPONSIBILITIES: {responsibilities}\n\n"
     "CANDIDATE: {name} - {headline}\n"
     "EXPERIENCE: {years} years ({experience_check} for this role) - {experience}\n"
     "ACHIEVEMENTS: {achievements}\n\n"
     "MATCHED REQUIRED SKILLS: {matched}\nMISSING REQUIRED SKILLS: {missing}\n"
     "MATCHED PREFERRED SKILLS: {matched_pref}"),
])

GAP_COACH = ChatPromptTemplate.from_messages([
    ("system",
     "You are a pragmatic mentor. For each missing skill, propose a small, concrete portfolio mini-project "
     "or resource that this specific candidate could finish quickly, building on skills they already have."),
    ("human",
     "TARGET ROLE: {title} at {company}\nCANDIDATE ALREADY KNOWS: {known}\nMISSING SKILLS: {missing}"),
])

INTERVIEW_COACH = ChatPromptTemplate.from_messages([
    ("system",
     "You are an interview coach. Predict exactly 6 questions this interviewer is likely to ask: "
     "3 technical questions on the role's core skills, 1 behavioural question, and 2 gap-probe questions "
     "targeting the candidate's weak areas. For each, give a hint on how THIS candidate should answer "
     "using their real experience."),
    ("human",
     "ROLE: {title} at {company}\nRESPONSIBILITIES: {responsibilities}\n"
     "CANDIDATE ACHIEVEMENTS: {achievements}\nCANDIDATE GAPS: {missing}"),
])

LETTER_WRITER = ChatPromptTemplate.from_messages([
    ("system",
     "You write concise, specific cover letters that recruiters actually read. Rules:\n"
     "- 180 to 300 words, 3-4 short paragraphs, no placeholders like [Address].\n"
     "- Open with a hook tied to something specific in the COMPANY CONTEXT, not 'I am writing to apply'.\n"
     "- Plain, confident language. Avoid: thrilled, excited, passionate, innovative, ideal candidate.\n"
     "- Cite 2-3 concrete achievements from the ACHIEVEMENTS list, copying their numbers exactly.\n"
     "- Use ONLY numbers, employers and project names that appear in the input. Never invent metrics.\n"
     "- Never claim skills the candidate does not have; frame gaps as things they are actively learning.\n"
     "- End with a confident call to action and sign off with the candidate's name.\n"
     "Output only the letter text."),
    ("human",
     "ROLE: {title} at {company}\nCOMPANY CONTEXT: {company_context}\nRESPONSIBILITIES: {responsibilities}\n"
     "CANDIDATE: {name} - {headline}\nEMPLOYMENT: {experience}\nACHIEVEMENTS: {achievements}\n"
     "STRENGTHS FOR THIS ROLE: {strengths}\nSKILLS THEY LACK: {missing}\n"
     "{revision_note}"),
])

LETTER_CRITIC = ChatPromptTemplate.from_messages([
    ("system",
     "You are a demanding senior recruiter reviewing a cover letter. Answer each rubric check honestly "
     "with true or false, judging only the letter in front of you. For every failed check, write one "
     "specific, actionable fix that quotes the offending sentence."),
    ("human",
     "ROLE: {title} at {company}\nSKILLS THE CANDIDATE LACKS (must not be claimed): {missing}\n\n"
     "COVER LETTER:\n{letter}"),
])
