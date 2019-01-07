---
layout: page
title: About
permalink: /about/
---

<ul class="post-list">
    <!-- 
        Here is the main paginator logic called.
        All calls to site.posts should be replaced by paginator.posts 
    -->
    {% for post in paginator.posts %}
      <li>
        <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>

        <h2>
          <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
        </h2>
        {%- if site.show_excerpts -%}
        {{ post.excerpt }}
      {%- endif -%}
      </li>
    {% endfor %}
  </ul>
  
  {% for post in paginator.posts %}
<div class="post-preview">
    <a href="{{ post.url | prepend: site.baseurl }}">
        <h2 class="post-title">
            {{ post.title }}
        </h2>
        {% if post.subtitle %}
        <h3 class="post-subtitle">
            {{ post.subtitle }}
        </h3>
        {% endif %}
        <div class="post-content-preview">
            {% if post.lang == 'en' %}
                {{ post.content | strip_html | truncate:300 }}
            {% else %}
                {{ post.content | strip_html | truncate:200 }}
            {% endif %}
        </div>
    </a>
    <p class="post-meta">
        Posted by {% if post.author %}{{ post.author }}{% else %}{{ site.title }}{% endif %} on {{ post.date | date: "%B %-d, %Y" }}
    </p>
</div>
<hr>
{% endfor %}
对于 Machine Learning 学习的一个记录。
