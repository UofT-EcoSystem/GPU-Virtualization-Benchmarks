import pandas as pd


table_style = [{'selector': 'tr:nth-of-type(odd)',
  'props': [('background', '#eee')]},
 {'selector': 'tr:nth-of-type(even)',
  'props': [('background', 'white')]},
 {'selector': 'th',
  'props': [('background', '#606060'),
            ('color', 'white'),
            ('font-family', 'verdana'),
            ("font-size", "90%")]},
 {'selector': 'td',
  'props': [('font-family', 'verdana'),
            ("font-size", "90%"),
            ("font-weight", "bold")]},
 {'selector': 'tr:hover',
  'props': [('background-color', '#ffffcc')]}
]
