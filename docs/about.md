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
  
对于 Machine Learning 学习的一个记录。
