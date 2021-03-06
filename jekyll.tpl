{% extends 'markdown.tpl' %}

{%- block header -%}
---
layout: post
title: {{nb.metadata['title']}}
date:   {{nb.metadata['modified_date']}}
author: sangjae kang
categories: {{nb.metadata['categories']}}
tags:	{{nb.metadata['tags']}}
use_math: true
---
{%- endblock header -%}

{% block in_prompt %}
{% endblock in_prompt %}

{% block input %}
<div class="input_area" markdown="1">
{{ super() }}
</div>
{% endblock input %}

{% block output_prompt %}
Output : 
{% endblock output_prompt %}

{% block stream %}
{:.output_stream}

```
{{ output.text }}
```
{% endblock stream %}

{% block data_text %}
{:.output_data_text}

```
{{ output.data['text/plain'] }}
```
{% endblock data_text %}

{% block traceback_line  %}
{:.output_traceback_line}

`{{ line | strip_ansi }}`

{% endblock traceback_line  %}

{% block data_html %}
<div markdown="0">
{{ output.data['text/html'] }}
</div>
{% endblock data_html %}
