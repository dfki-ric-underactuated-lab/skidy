import os
from os.path import dirname
from pathlib import Path

import pdoc
from pdoc import extract, render, doc

import skidy


def custom_pdoc(
    *modules: Path,
    output_directory: Path=None,
    ) -> None:
    """
    Render the documentation for a list of modules.
     - If `output_directory` is `None`, returns the rendered documentation
       for the first module in the list.
     - If `output_directory` is set, recursively writes the rendered output
       for all specified modules and their submodules to the target destination.
    Rendering options can be configured by calling `pdoc.render.configure` in advance.
    """
    all_modules: dict[str, doc.Module] = {}
    for module_name in extract.walk_specs(modules):
        all_modules[module_name] = doc.Module.from_name(module_name)
    all_modules["skidy"].docstring += "\n.. include:: ../../README.md"
    for module in all_modules.values():
        out = render.html_module(module, all_modules)
        if not output_directory:
            return out
        else:
            outfile = output_directory / f"{module.fullname.replace('.', '/')}.html"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(out.encode())

    assert output_directory

    index = render.html_index(all_modules)
    if index:
        (output_directory / "index.html").write_bytes(index.encode())

    search = render.search_index(all_modules)
    if search:
        (output_directory / "search.js").write_bytes(search.encode())

    return None

def main():
    pdoc.render.configure(
        docformat="google",
        favicon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1cqs1ylwsW2UUHzulEp2_K3aKcg2KDIVJal-v0ZNbtg&s",
        footer_text=f"skidy {skidy.__version__}",
        logo="https://www.dfki.de/fileadmin/user_upload/DFKI/Medien/Logos/Logos_DFKI/DFKI_Logo.png",
        logo_link=None,        
    )
    
    custom_pdoc(
        Path(os.path.join(dirname(dirname(__file__)),"src", "skidy")),
        output_directory=Path(os.path.join(dirname(__file__),"html")),
    )
    



if __name__ == "__main__":
    main()