PROMPT_MUSIC_GEN = [
    """
    You are an expert in creating prompts for AI models for music generation. Your task is to write a concise prompt of 10-20 words to describe a piece of music based on the provided information.

    The name of the music piece is:
    <name>
    {NAME}
    </name>

    The metadata of the music piece is:
    <metadata>
    {METADATA}
    </metadata>

    A no-accurate description of the music for each 10-second slice is provided below:
    <claps>
    {CLAPS}
    </claps>

    <thinkingstep>
    Carefully analyze the name, metadata, and the no-accurate description of the music piece. Consider the key elements, such as the genre, mood, instruments, and the progression of the music over time. Identify the most important aspects that capture the essence of the music piece.
    </thinkingstep>

    Based on your analysis, write a concise and descriptive prompt of 10-20 words that encapsulates the core characteristics of the music piece. Output your prompt inside <prompt> tags.
    </thinkingstep>
    """.strip(),
]


PROMPT_TTS = ["hello world this is a test"]

PROMPT_REMIX = ["hello world this is a test"]

PROMPT_LYRICS = ["hello world this is a test"]

PROMPT_LST = {1: PROMPT_MUSIC_GEN.copy(), 2: PROMPT_TTS.copy(), 3: PROMPT_REMIX.copy}


PROMPT_FORMAT_LYRICS = [
    """
    Your are an expert in creating prompts for AI models for tagging.
    Your task is to label the different parts of these song lyrics by adding the appropriate tag in square brackets at the start of each section. 

    The tags you should use are:
    - [VERSE] 
    - [CHORUS]
    - [BRIDGE]

    For example, if a section of the lyrics is a verse, put [VERSE] at the start of the first line of that section.  Do not add too many label to the output. Write the complete lyric in the output, never remove any part of the lyric in the output.
    Please make sure to separate each labeled section by a blank line. Do not write anything else besides the lyrics with the labels added. The output should just be the labeled lyrics with blank lines between sections.

    Here are the lyrics to a song:
    {LYRICS}
    """.strip(),
]

# the instruction for the phi3 model
INSTRUCT_PHI3 = {
    "role": "system",
    "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
}
