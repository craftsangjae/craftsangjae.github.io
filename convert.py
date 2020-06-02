import os
import nbformat
from datetime import datetime, timedelta
from nbconvert import MarkdownExporter
import click

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.argument('filename', type=click.Path(exists=True))
def convert_notebook_to_markdown(filename):
    global ROOT_DIR
    with open(filename,'r') as f:
        notebook = nbformat.read(f, 4)        
        
    # Metadata 설정하기
    fname = os.path.splitext(
        os.path.split(filename)[1])[0]
    metadata = [name.strip() for name in fname.split('|')]
    
    notebook.metadata['title'] = metadata[0]
    if len(metadata) >= 2:
        notebook.metadata['categories'] = metadata[1].strip()
    if len(metadata) >= 3:
        notebook.metadata['tags'] = metadata[2].strip()    
    
    curr_time = datetime.now()-timedelta(days=1)
    notebook.metadata['modified_date'] = str(curr_time)[:-7]

    exporter = MarkdownExporter()
    exporter.template_file = os.path.join(ROOT_DIR, 'jekyll.tpl')
    body, resource = exporter.from_notebook_node(notebook)
    
    curr_date = curr_time.strftime("%Y-%m-%d")
    markdown_path = os.path.join(
        ROOT_DIR, "_posts", curr_date+"-"+metadata[0]+".md")
    
    with open(markdown_path, "w") as f:
        f.write(body)


if __name__ == "__main__":
	convert_notebook_to_markdown()