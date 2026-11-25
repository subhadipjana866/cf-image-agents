"""Image-generation and image-edit agents, each a LangChain tool-calling agent.

    from graphic_designer.agents import Brief, DesignSession, build_edit_agent
    session = DesignSession()
    session.start_with_image("https://.../poster.webp")
    agent = build_edit_agent()
    out = agent.invoke({"messages": [("user", "background navy, then square")]}, context=session)

Test from the shell: ``python -m graphic_designer.agents --help``.
"""
from .build import build_edit_agent, build_generation_agent
from .session import Brief, DesignSession, Version

__all__ = ["Brief", "DesignSession", "Version", "build_edit_agent", "build_generation_agent"]
