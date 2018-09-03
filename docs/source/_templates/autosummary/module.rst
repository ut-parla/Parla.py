{{ fullname | escape | underline}}

.. py:module:: {{ fullname }}


{% block functions %}
{% if functions %}
Functions
---------

{% for item in functions %}
.. autofunction:: {{ item }}
                  
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
{% for item in classes %}
{{ item | escape | underline(line="-")}}

.. autoclass:: {{ item }}
               
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
Exceptions
----------

{% for item in exceptions %}
.. autoexception:: {{ item }}
                   
{%- endfor %}
{% endif %}
{% endblock %}
