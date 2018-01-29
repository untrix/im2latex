

```python
from IPython.core.display import display, Latex, Pretty, Math, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import sys
import pandas as pd
sys.path.extend(['../src/commons'])
import pub_commons as pub
%matplotlib inline
```


<style>.container { width:90% !important; }</style>



```python
pub.disp_rand_sample('I2L-NOPOOL')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>edit_distance</th>
      <th>$\mathbf{y}$_len</th>
      <th>$\mathbf{y}$</th>
      <th>$\mathbf{\hat{y}}$_len</th>
      <th>$\mathbf{\hat{y}}$</th>
      <th>$\mathbf{y}$_seq</th>
      <th>$\mathbf{\hat{y}}$_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.169492</td>
      <td>59</td>
      <td>$E ( z + { n } \cdot { \alpha } + { m } \cdot { \beta } , z ) = e ^ { - \pi i m \cdot \Omega \cdot m - 2 \pi i m \cdot \left( I ( z ) - I ( w ) \right) } E ( z , w ) . $</td>
      <td>51</td>
      <td>$E ( z + n \cdot \alpha + m \cdot \beta , z ) = e ^ { - \pi i m \cdot \Omega \cdot m - 2 \pi i m \cdot ( I ( z ) - I ( w ) ) } E ( z , w ) . $</td>
      <td>E ( z + { n } \cdot { \alpha } + { m } \cdot { \beta } , z ) = e ^ { - \pi i m \cdot \Omega \cdot m - 2 \pi i m \cdot \left( I ( z ) - I ( w ) \right) } E ( z , w ) .</td>
      <td>E ( z + n \cdot \alpha + m \cdot \beta , z ) = e ^ { - \pi i m \cdot \Omega \cdot m - 2 \pi i m \cdot ( I ( z ) - I ( w ) ) } E ( z , w ) .</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.016393</td>
      <td>122</td>
      <td>$\Gamma ( b \rightarrow J / \psi + X ) = \frac { G _ { F } ^ { 2 } } { 1 4 4 \pi } | V _ { c b } | ^ { 2 } m _ { c } m _ { b } ^ { 3 } \left( 1 - \frac { 4 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) ^ { 2 } \left[ a \left( 1 + \frac { 8 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) + b \right] \, , $</td>
      <td>121</td>
      <td>$\Gamma ( b \to J / \psi + X ) = \frac { G _ { F } ^ { 2 } } { 1 4 4 \pi } | V _ { c b } | ^ { 2 } m _ { c } m _ { b } ^ { 3 } \left( 1 - \frac { 4 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) ^ { 2 } \left[ a \left( 1 + \frac { 8 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) + b \right] , $</td>
      <td>\Gamma ( b \rightarrow J / \psi + X ) = \frac { G _ { F } ^ { 2 } } { 1 4 4 \pi } | V _ { c b } | ^ { 2 } m _ { c } m _ { b } ^ { 3 } \left( 1 - \frac { 4 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) ^ { 2 } \left[ a \left( 1 + \frac { 8 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) + b \right] \, ,</td>
      <td>\Gamma ( b \to J / \psi + X ) = \frac { G _ { F } ^ { 2 } } { 1 4 4 \pi } | V _ { c b } | ^ { 2 } m _ { c } m _ { b } ^ { 3 } \left( 1 - \frac { 4 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) ^ { 2 } \left[ a \left( 1 + \frac { 8 m _ { c } ^ { 2 } } { m _ { b } ^ { 2 } } \right) + b \right] ,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>73</td>
      <td>$S _ { c } ^ { ( 1 ) &gt; } ( x ^ { 0 } - i \beta - y ^ { 0 } , \vec { p } ) = - e ^ { - \beta \mu } S _ { c } ^ { ( 1 ) &lt; } ( x ^ { 0 } - y ^ { 0 } , \vec { p } ) . $</td>
      <td>73</td>
      <td>$S _ { c } ^ { ( 1 ) &gt; } ( x ^ { 0 } - i \beta - y ^ { 0 } , \vec { p } ) = - e ^ { - \beta \mu } S _ { c } ^ { ( 1 ) &lt; } ( x ^ { 0 } - y ^ { 0 } , \vec { p } ) . $</td>
      <td>S _ { c } ^ { ( 1 ) &gt; } ( x ^ { 0 } - i \beta - y ^ { 0 } , \vec { p } ) = - e ^ { - \beta \mu } S _ { c } ^ { ( 1 ) &lt; } ( x ^ { 0 } - y ^ { 0 } , \vec { p } ) .</td>
      <td>S _ { c } ^ { ( 1 ) &gt; } ( x ^ { 0 } - i \beta - y ^ { 0 } , \vec { p } ) = - e ^ { - \beta \mu } S _ { c } ^ { ( 1 ) &lt; } ( x ^ { 0 } - y ^ { 0 } , \vec { p } ) .</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.182540</td>
      <td>126</td>
      <td>$\operatorname { l n } \: \frac { E _ { m a x } } { m _ { h } } \; \approx \; \frac { 1 } { 2 } \: \operatorname { l n } \: \frac { E } { m _ { h } \operatorname { s i n } \Theta } \: - \: B \: \left( \sqrt { \frac { b } { 1 6 N _ { C } } \: Y _ { \Theta } } \; - \; \sqrt { \frac { b } { 1 6 N _ { C } } \; \operatorname { l n } \: \frac { m _ { h } } { \Lambda } \: } \right) . $</td>
      <td>125</td>
      <td>$\operatorname { l n } \, \frac { m _ { a m c } } { m _ { h } } \, \approx \, \frac { 1 } { 2 } \, \operatorname { l n } \frac { E } { m _ { h } \operatorname { s i n } \Theta } \, - \, B \, \left( \sqrt { \frac { b } { 1 6 N _ { C } } } \, Y _ { \Theta } \ - \, \sqrt { \frac { b } { 1 6 N _ { C } } } \, \operatorname { l n } \, \frac { m _ { h } } { \Lambda } \right) \, . $</td>
      <td>\operatorname { l n } \: \frac { E _ { m a x } } { m _ { h } } \; \approx \; \frac { 1 } { 2 } \: \operatorname { l n } \: \frac { E } { m _ { h } \operatorname { s i n } \Theta } \: - \: B \: \left( \sqrt { \frac { b } { 1 6 N _ { C } } \: Y _ { \Theta } } \; - \; \sqrt { \frac { b } { 1 6 N _ { C } } \; \operatorname { l n } \: \frac { m _ { h } } { \Lambda } \: } \right) .</td>
      <td>\operatorname { l n } \, \frac { m _ { a m c } } { m _ { h } } \, \approx \, \frac { 1 } { 2 } \, \operatorname { l n } \frac { E } { m _ { h } \operatorname { s i n } \Theta } \, - \, B \, \left( \sqrt { \frac { b } { 1 6 N _ { C } } } \, Y _ { \Theta } \ - \, \sqrt { \frac { b } { 1 6 N _ { C } } } \, \operatorname { l n } \, \frac { m _ { h } } { \Lambda } \right) \, .</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.014085</td>
      <td>71</td>
      <td>$\tilde { A } _ { i } \rightarrow \epsilon _ { i j } \biggl \{ - \alpha \frac { x ^ { j } - q ^ { j } } { | \vec { x } - \vec { q } | ^ { 2 } } + \frac { \rho _ { e } } { 2 \kappa } q ^ { j } \biggr \} , $</td>
      <td>71</td>
      <td>$\tilde { A } _ { i } \rightarrow \epsilon _ { i j } \biggl \{ - \alpha \frac { x ^ { j } - q ^ { j } } { | \vec { x } - \vec { q } ] ^ { 2 } } + \frac { \rho _ { e } } { 2 \kappa } q ^ { j } \biggr \} , $</td>
      <td>\tilde { A } _ { i } \rightarrow \epsilon _ { i j } \biggl \{ - \alpha \frac { x ^ { j } - q ^ { j } } { | \vec { x } - \vec { q } | ^ { 2 } } + \frac { \rho _ { e } } { 2 \kappa } q ^ { j } \biggr \} ,</td>
      <td>\tilde { A } _ { i } \rightarrow \epsilon _ { i j } \biggl \{ - \alpha \frac { x ^ { j } - q ^ { j } } { | \vec { x } - \vec { q } ] ^ { 2 } } + \frac { \rho _ { e } } { 2 \kappa } q ^ { j } \biggr \} ,</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008850</td>
      <td>113</td>
      <td>$\omega _ { 0 } ( x ) = 2 \int _ { 0 } ^ { \infty } { \frac { x \rho ( \lambda ) d \lambda } { x ^ { 2 } - \lambda ^ { 2 } } } = \int _ { 0 } ^ { \infty } \rho ( \lambda ) d \lambda \left( { \frac { 1 } { x - \lambda } } + { \frac { 1 } { x + \lambda } } \right) = \int _ { - \infty } ^ { \infty } { \frac { \rho ( \lambda ) d \lambda } { x - \lambda } } , $</td>
      <td>113</td>
      <td>$\omega _ { 0 } ( x ) = 2 \int _ { 0 } ^ { \infty } { \frac { x \rho ( \lambda ) d \lambda } { x ^ { 2 } - \lambda ^ { 2 } } } = \int _ { 0 } ^ { \infty } \rho ( \lambda ) d \lambda \left( { \frac { 1 } { x - \lambda } } + { \frac { 1 } { x + \lambda } } \right) = \int _ { - \infty } ^ { \infty } { \frac { \rho ( \lambda ) - \lambda } { x - \lambda } } , $</td>
      <td>\omega _ { 0 } ( x ) = 2 \int _ { 0 } ^ { \infty } { \frac { x \rho ( \lambda ) d \lambda } { x ^ { 2 } - \lambda ^ { 2 } } } = \int _ { 0 } ^ { \infty } \rho ( \lambda ) d \lambda \left( { \frac { 1 } { x - \lambda } } + { \frac { 1 } { x + \lambda } } \right) = \int _ { - \infty } ^ { \infty } { \frac { \rho ( \lambda ) d \lambda } { x - \lambda } } ,</td>
      <td>\omega _ { 0 } ( x ) = 2 \int _ { 0 } ^ { \infty } { \frac { x \rho ( \lambda ) d \lambda } { x ^ { 2 } - \lambda ^ { 2 } } } = \int _ { 0 } ^ { \infty } \rho ( \lambda ) d \lambda \left( { \frac { 1 } { x - \lambda } } + { \frac { 1 } { x + \lambda } } \right) = \int _ { - \infty } ^ { \infty } { \frac { \rho ( \lambda ) - \lambda } { x - \lambda } } ,</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.105263</td>
      <td>57</td>
      <td>$\left[ i D ^ { \mu } , i D ^ { \nu } \right] h _ { v } ^ { ( \pm ) } \quad \ll \quad \left\{ i D ^ { \mu } , i D ^ { \nu } \right\} h _ { v } ^ { ( \pm ) } \, . $</td>
      <td>59</td>
      <td>$[ i D ^ { \mu } , i D ^ { \nu } ] \, h _ { v } ^ { ( \pm ) } \quad \ll \quad \{ i D ^ { \mu } , i D ^ { \nu } \} \, h _ { v } ^ { ( \pm ) } \, . $</td>
      <td>\left[ i D ^ { \mu } , i D ^ { \nu } \right] h _ { v } ^ { ( \pm ) } \quad \ll \quad \left\{ i D ^ { \mu } , i D ^ { \nu } \right\} h _ { v } ^ { ( \pm ) } \, .</td>
      <td>[ i D ^ { \mu } , i D ^ { \nu } ] \, h _ { v } ^ { ( \pm ) } \quad \ll \quad \{ i D ^ { \mu } , i D ^ { \nu } \} \, h _ { v } ^ { ( \pm ) } \, .</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.119048</td>
      <td>42</td>
      <td>$\Delta _ { 1 } \supset \Delta _ { 2 } , \ \ \mathrm { a n d } \ \ \Delta _ { 1 } ^ { * } \subset \Delta _ { 2 } ^ { * } . $</td>
      <td>42</td>
      <td>$\Delta _ { 1 } \circ \Delta _ { 2 } , ~ ~ \mathrm { a n d } ~ ~ \Delta _ { 1 } ^ { * } \subset \Delta _ { 2 } ^ { * } . $</td>
      <td>\Delta _ { 1 } \supset \Delta _ { 2 } , \ \ \mathrm { a n d } \ \ \Delta _ { 1 } ^ { * } \subset \Delta _ { 2 } ^ { * } .</td>
      <td>\Delta _ { 1 } \circ \Delta _ { 2 } , ~ ~ \mathrm { a n d } ~ ~ \Delta _ { 1 } ^ { * } \subset \Delta _ { 2 } ^ { * } .</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
      <td>74</td>
      <td>$[ ( \gamma _ { 4 } P _ { 0 } - \gamma _ { i } P _ { i } ) + \frac { \epsilon } { 2 } ( \gamma _ { 4 } ( P _ { 0 } ^ { 2 } - P _ { i } P _ { i } ) - m P _ { 0 } ) ] \psi = m \psi . $</td>
      <td>74</td>
      <td>$[ ( \gamma _ { 4 } P _ { 0 } - \gamma _ { i } P _ { i } ) + \frac { \epsilon } { 2 } ( \gamma _ { 4 } ( P _ { 0 } ^ { 2 } - P _ { i } P _ { i } ) - m P _ { 0 } ) ] \psi = m \psi . $</td>
      <td>[ ( \gamma _ { 4 } P _ { 0 } - \gamma _ { i } P _ { i } ) + \frac { \epsilon } { 2 } ( \gamma _ { 4 } ( P _ { 0 } ^ { 2 } - P _ { i } P _ { i } ) - m P _ { 0 } ) ] \psi = m \psi .</td>
      <td>[ ( \gamma _ { 4 } P _ { 0 } - \gamma _ { i } P _ { i } ) + \frac { \epsilon } { 2 } ( \gamma _ { 4 } ( P _ { 0 } ^ { 2 } - P _ { i } P _ { i } ) - m P _ { 0 } ) ] \psi = m \psi .</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.086420</td>
      <td>81</td>
      <td>$\phi ^ { \prime \prime } - { \frac { 5 \alpha + 1 0 } { 8 } } k \phi ^ { \prime } + \left( m ^ { 2 } - { \frac { l ^ { 2 } } { R ^ { 2 } } } e ^ { { \frac { 5 } { 4 } } ( \alpha - 2 ) k \rho } \right) e ^ { k \rho } \phi = 0 $</td>
      <td>75</td>
      <td>$\phi ^ { \prime \prime } - \frac { 5 \alpha + 1 0 } { 8 } k \phi ^ { \prime } + \left( m ^ { 2 } - \frac { l ^ { 2 } } { R ^ { 2 } } e ^ { \frac { i } { 4 } ( \alpha - 2 ) k \rho } \right) e ^ { k \rho } \phi = 0 $</td>
      <td>\phi ^ { \prime \prime } - { \frac { 5 \alpha + 1 0 } { 8 } } k \phi ^ { \prime } + \left( m ^ { 2 } - { \frac { l ^ { 2 } } { R ^ { 2 } } } e ^ { { \frac { 5 } { 4 } } ( \alpha - 2 ) k \rho } \right) e ^ { k \rho } \phi = 0</td>
      <td>\phi ^ { \prime \prime } - \frac { 5 \alpha + 1 0 } { 8 } k \phi ^ { \prime } + \left( m ^ { 2 } - \frac { l ^ { 2 } } { R ^ { 2 } } e ^ { \frac { i } { 4 } ( \alpha - 2 ) k \rho } \right) e ^ { k \rho } \phi = 0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.122642</td>
      <td>106</td>
      <td>$\alpha _ { q } = \sqrt { \frac { \operatorname { s i n h } \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \operatorname { s i n h } \lambda } } \alpha , ~ ~ ~ ~ ~ ~ ~ \alpha _ { q } ^ { * } = \sqrt { \frac { \operatorname { s i n h } \, \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \operatorname { s i n h } \lambda } } \alpha ^ { * } . $</td>
      <td>102</td>
      <td>$\alpha _ { q } = \sqrt { \frac { \mathrm { s i n h } \, \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \mathrm { s i n h } \lambda } } \alpha , \qquad \alpha _ { q } ^ { * } = \sqrt { \frac { \mathrm { s i n h } \, \lambda \alpha \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \mathrm { s i n h } \lambda } } \alpha ^ { * } . $</td>
      <td>\alpha _ { q } = \sqrt { \frac { \operatorname { s i n h } \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \operatorname { s i n h } \lambda } } \alpha , ~ ~ ~ ~ ~ ~ ~ \alpha _ { q } ^ { * } = \sqrt { \frac { \operatorname { s i n h } \, \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \operatorname { s i n h } \lambda } } \alpha ^ { * } .</td>
      <td>\alpha _ { q } = \sqrt { \frac { \mathrm { s i n h } \, \lambda \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \mathrm { s i n h } \lambda } } \alpha , \qquad \alpha _ { q } ^ { * } = \sqrt { \frac { \mathrm { s i n h } \, \lambda \alpha \alpha \alpha ^ { * } } { \alpha \alpha ^ { * } \, \mathrm { s i n h } \lambda } } \alpha ^ { * } .</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.035714</td>
      <td>56</td>
      <td>$\Gamma _ { A _ { n } } ^ { \pm } \, \equiv \, p _ { a _ { n } } \, \pm \, \frac { \omega _ { n + \frac { 1 } { 2 } } } { 2 R } b _ { n } \approx 0 , $</td>
      <td>56</td>
      <td>$\Gamma _ { A _ { n } } ^ { \pm } \; \equiv \; p _ { a _ { n } } \, \pm \, \frac { \omega _ { n + \frac { 1 } { 2 } } } { 2 R } b _ { n } \approx 0 , $</td>
      <td>\Gamma _ { A _ { n } } ^ { \pm } \, \equiv \, p _ { a _ { n } } \, \pm \, \frac { \omega _ { n + \frac { 1 } { 2 } } } { 2 R } b _ { n } \approx 0 ,</td>
      <td>\Gamma _ { A _ { n } } ^ { \pm } \; \equiv \; p _ { a _ { n } } \, \pm \, \frac { \omega _ { n + \frac { 1 } { 2 } } } { 2 R } b _ { n } \approx 0 ,</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.013158</td>
      <td>76</td>
      <td>$n ^ { i } = ( S ^ { t } ) _ { j } ^ { i } e ^ { j } - ( N ^ { t } ) ^ { i j } \sum _ { r = 1 } ^ { d - 1 } \sum _ { \bf k } k _ { j } N _ { r } ( { \bf k } ) \, , $</td>
      <td>75</td>
      <td>$n ^ { i } = ( S ^ { t } ) _ { j } ^ { i } e ^ { j } - ( N ^ { t } ) ^ { i j } \sum _ { r = 1 } ^ { d - 1 } \sum _ { k } k _ { j } N _ { r } ( { \bf k } ) \, , $</td>
      <td>n ^ { i } = ( S ^ { t } ) _ { j } ^ { i } e ^ { j } - ( N ^ { t } ) ^ { i j } \sum _ { r = 1 } ^ { d - 1 } \sum _ { \bf k } k _ { j } N _ { r } ( { \bf k } ) \, ,</td>
      <td>n ^ { i } = ( S ^ { t } ) _ { j } ^ { i } e ^ { j } - ( N ^ { t } ) ^ { i j } \sum _ { r = 1 } ^ { d - 1 } \sum _ { k } k _ { j } N _ { r } ( { \bf k } ) \, ,</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000000</td>
      <td>42</td>
      <td>$E _ { c } = - 2 \frac { n l r _ { + } ^ { n - 1 } V o l ( \Sigma _ { n } ) } { 1 6 \pi G R } . $</td>
      <td>42</td>
      <td>$E _ { c } = - 2 \frac { n l r _ { + } ^ { n - 1 } V o l ( \Sigma _ { n } ) } { 1 6 \pi G R } . $</td>
      <td>E _ { c } = - 2 \frac { n l r _ { + } ^ { n - 1 } V o l ( \Sigma _ { n } ) } { 1 6 \pi G R } .</td>
      <td>E _ { c } = - 2 \frac { n l r _ { + } ^ { n - 1 } V o l ( \Sigma _ { n } ) } { 1 6 \pi G R } .</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.024390</td>
      <td>41</td>
      <td>$- i \int d \theta \lambda _ { a } ^ { \dag } ( A _ { \alpha } ) ^ { a b } \lambda _ { b } \partial _ { \theta } X ^ { \alpha } $</td>
      <td>41</td>
      <td>$- i \int d \theta \lambda _ { a } ^ { \dagger } ( A _ { \alpha } ) ^ { a b } \lambda _ { b } \partial _ { \theta } X ^ { \alpha } $</td>
      <td>- i \int d \theta \lambda _ { a } ^ { \dag } ( A _ { \alpha } ) ^ { a b } \lambda _ { b } \partial _ { \theta } X ^ { \alpha }</td>
      <td>- i \int d \theta \lambda _ { a } ^ { \dagger } ( A _ { \alpha } ) ^ { a b } \lambda _ { b } \partial _ { \theta } X ^ { \alpha }</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.000000</td>
      <td>15</td>
      <td>$\nabla ^ { \mu } T _ { \mu \nu } = 0 \, . $</td>
      <td>15</td>
      <td>$\nabla ^ { \mu } T _ { \mu \nu } = 0 \, . $</td>
      <td>\nabla ^ { \mu } T _ { \mu \nu } = 0 \, .</td>
      <td>\nabla ^ { \mu } T _ { \mu \nu } = 0 \, .</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.000000</td>
      <td>32</td>
      <td>$\varphi ^ { \mu } { } _ { i \alpha } = \hat { D } _ { i } \varphi _ { \alpha } ^ { \mu } \, , $</td>
      <td>32</td>
      <td>$\varphi ^ { \mu } { } _ { i \alpha } = \hat { D } _ { i } \varphi _ { \alpha } ^ { \mu } \, , $</td>
      <td>\varphi ^ { \mu } { } _ { i \alpha } = \hat { D } _ { i } \varphi _ { \alpha } ^ { \mu } \, ,</td>
      <td>\varphi ^ { \mu } { } _ { i \alpha } = \hat { D } _ { i } \varphi _ { \alpha } ^ { \mu } \, ,</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.166667</td>
      <td>30</td>
      <td>$y \propto \left[ \operatorname { c o t } ( \eta / 2 ) \right] ^ { ( 1 + \sqrt { 1 3 } ) / 2 } , $</td>
      <td>30</td>
      <td>$y \propto [ \mathrm { c o t } ( \eta / 2 ) ] ^ { ( 1 + \sqrt { 3 } ) / 2 } \, , $</td>
      <td>y \propto \left[ \operatorname { c o t } ( \eta / 2 ) \right] ^ { ( 1 + \sqrt { 1 3 } ) / 2 } ,</td>
      <td>y \propto [ \mathrm { c o t } ( \eta / 2 ) ] ^ { ( 1 + \sqrt { 3 } ) / 2 } \, ,</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.000000</td>
      <td>23</td>
      <td>$\{ a _ { \mu } , a _ { \nu } \} = - \delta _ { \mu \nu } \, , $</td>
      <td>23</td>
      <td>$\{ a _ { \mu } , a _ { \nu } \} = - \delta _ { \mu \nu } \, , $</td>
      <td>\{ a _ { \mu } , a _ { \nu } \} = - \delta _ { \mu \nu } \, ,</td>
      <td>\{ a _ { \mu } , a _ { \nu } \} = - \delta _ { \mu \nu } \, ,</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.125000</td>
      <td>56</td>
      <td>$\left. \partial _ { 0 } \phi ( x _ { 0 } , { \bf x } ) \right| _ { x _ { 0 } = \epsilon } \to \epsilon ^ { - \lambda } \partial _ { \epsilon } \phi _ { h } ( \epsilon , { \bf x } ) . $</td>
      <td>55</td>
      <td>$\partial _ { 0 } \phi ( x _ { 0 } , \mathbf { x } ) | _ { x _ { 0 } = \epsilon } \rightarrow \epsilon ^ { - \lambda } \partial _ { \epsilon } \phi _ { h } ( \epsilon , \mathbf { x } ) . $</td>
      <td>\left. \partial _ { 0 } \phi ( x _ { 0 } , { \bf x } ) \right| _ { x _ { 0 } = \epsilon } \to \epsilon ^ { - \lambda } \partial _ { \epsilon } \phi _ { h } ( \epsilon , { \bf x } ) .</td>
      <td>\partial _ { 0 } \phi ( x _ { 0 } , \mathbf { x } ) | _ { x _ { 0 } = \epsilon } \rightarrow \epsilon ^ { - \lambda } \partial _ { \epsilon } \phi _ { h } ( \epsilon , \mathbf { x } ) .</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.076923</td>
      <td>78</td>
      <td>$G _ { N } { \frac { | \hat { h } _ { m } ( 0 ) | ^ { 2 } } { e _ { l } } } \sim { \frac { G _ { N } } { y _ { * } } } \approx { \frac { 1 0 ^ { 1 1 } } { M _ { * } ^ { 2 } } } \, . $</td>
      <td>72</td>
      <td>$G _ { N } \frac { | \hat { h } _ { m } ( 0 ) | ^ { 2 } } { e _ { l } } \sim \frac { G _ { N } } { y _ { * } } \approx \frac { 1 0 ^ { 1 1 } } { M _ { * } ^ { 2 } } \, . $</td>
      <td>G _ { N } { \frac { | \hat { h } _ { m } ( 0 ) | ^ { 2 } } { e _ { l } } } \sim { \frac { G _ { N } } { y _ { * } } } \approx { \frac { 1 0 ^ { 1 1 } } { M _ { * } ^ { 2 } } } \, .</td>
      <td>G _ { N } \frac { | \hat { h } _ { m } ( 0 ) | ^ { 2 } } { e _ { l } } \sim \frac { G _ { N } } { y _ { * } } \approx \frac { 1 0 ^ { 1 1 } } { M _ { * } ^ { 2 } } \, .</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.000000</td>
      <td>52</td>
      <td>$T ^ { 3 D } \, = \, - 2 i \pi \, \tau ^ { 2 } \, \beta _ { 1 } \beta _ { 2 } \, T ( s _ { 1 } , s _ { 1 } ) \, \tau ^ { 2 } . $</td>
      <td>52</td>
      <td>$T ^ { 3 D } \, = \, - 2 i \pi \, \tau ^ { 2 } \, \beta _ { 1 } \beta _ { 2 } \, T ( s _ { 1 } , s _ { 1 } ) \, \tau ^ { 2 } . $</td>
      <td>T ^ { 3 D } \, = \, - 2 i \pi \, \tau ^ { 2 } \, \beta _ { 1 } \beta _ { 2 } \, T ( s _ { 1 } , s _ { 1 } ) \, \tau ^ { 2 } .</td>
      <td>T ^ { 3 D } \, = \, - 2 i \pi \, \tau ^ { 2 } \, \beta _ { 1 } \beta _ { 2 } \, T ( s _ { 1 } , s _ { 1 } ) \, \tau ^ { 2 } .</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.000000</td>
      <td>38</td>
      <td>$K Q ^ { \dagger } e ^ { V } Q | _ { D } \rightarrow m _ { Q } ^ { 2 } | A _ { Q } | ^ { 2 } $</td>
      <td>38</td>
      <td>$K Q ^ { \dagger } e ^ { V } Q | _ { D } \rightarrow m _ { Q } ^ { 2 } | A _ { Q } | ^ { 2 } $</td>
      <td>K Q ^ { \dagger } e ^ { V } Q | _ { D } \rightarrow m _ { Q } ^ { 2 } | A _ { Q } | ^ { 2 }</td>
      <td>K Q ^ { \dagger } e ^ { V } Q | _ { D } \rightarrow m _ { Q } ^ { 2 } | A _ { Q } | ^ { 2 }</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.081081</td>
      <td>37</td>
      <td>$\mathrm { { \bf Q } } = \mathrm { d i a g } ( q _ { 1 } , q _ { 2 } , \cdots , q _ { N } ) . $</td>
      <td>34</td>
      <td>${ \bf Q } = \mathrm { d i a g } ( q _ { 1 } , q _ { 2 } , \cdots , q _ { N } ) . $</td>
      <td>\mathrm { { \bf Q } } = \mathrm { d i a g } ( q _ { 1 } , q _ { 2 } , \cdots , q _ { N } ) .</td>
      <td>{ \bf Q } = \mathrm { d i a g } ( q _ { 1 } , q _ { 2 } , \cdots , q _ { N } ) .</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.000000</td>
      <td>74</td>
      <td>$i \Sigma ^ { \pi N } ( p ^ { 2 } ) = 8 G ^ { 2 } [ p _ { \mu } I _ { l i n } ^ { \mu } ( p ^ { 2 } , m ^ { 2 } ) - I _ { q u a d } ( p ^ { 2 } , m ^ { 2 } ) ] $</td>
      <td>74</td>
      <td>$i \Sigma ^ { \pi N } ( p ^ { 2 } ) = 8 G ^ { 2 } [ p _ { \mu } I _ { l i n } ^ { \mu } ( p ^ { 2 } , m ^ { 2 } ) - I _ { q u a d } ( p ^ { 2 } , m ^ { 2 } ) ] $</td>
      <td>i \Sigma ^ { \pi N } ( p ^ { 2 } ) = 8 G ^ { 2 } [ p _ { \mu } I _ { l i n } ^ { \mu } ( p ^ { 2 } , m ^ { 2 } ) - I _ { q u a d } ( p ^ { 2 } , m ^ { 2 } ) ]</td>
      <td>i \Sigma ^ { \pi N } ( p ^ { 2 } ) = 8 G ^ { 2 } [ p _ { \mu } I _ { l i n } ^ { \mu } ( p ^ { 2 } , m ^ { 2 } ) - I _ { q u a d } ( p ^ { 2 } , m ^ { 2 } ) ]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.040000</td>
      <td>25</td>
      <td>$\rho ( s ) \propto \frac { s } { \operatorname { t a n h } ( 2 \pi s ) } ~ . $</td>
      <td>25</td>
      <td>$\rho ( s ) \propto \frac { s } { \operatorname { t a n h } ( 2 \pi s ) } \ . $</td>
      <td>\rho ( s ) \propto \frac { s } { \operatorname { t a n h } ( 2 \pi s ) } ~ .</td>
      <td>\rho ( s ) \propto \frac { s } { \operatorname { t a n h } ( 2 \pi s ) } \ .</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.121495</td>
      <td>107</td>
      <td>$z _ { 1 } ^ { ( A ) } \simeq 1 - { \frac { E _ { m i n } } { M } } - { \frac { | \vec { P } _ { A - 1 } | ^ { 2 } } { 2 M M _ { A - 1 } } } + \frac { \eta } { M } | \vec { P } _ { A - 1 } | c o s \theta _ { \widehat { \vec { P } _ { A - 1 } \vec { q } } } ~ $</td>
      <td>104</td>
      <td>$z _ { 1 } ^ { ( A ) } \simeq 1 - { \frac { E _ { m i n } } { M } } - \frac { | \vec { P } _ { A - 1 } | ^ { 2 } } { 2 M A _ { A - 1 } } + \frac { \eta } { M } | \vec { P } _ { A - 1 } | c o s \theta _ { \tilde { f } _ { A - \tilde { q } } \vec { q } } $</td>
      <td>z _ { 1 } ^ { ( A ) } \simeq 1 - { \frac { E _ { m i n } } { M } } - { \frac { | \vec { P } _ { A - 1 } | ^ { 2 } } { 2 M M _ { A - 1 } } } + \frac { \eta } { M } | \vec { P } _ { A - 1 } | c o s \theta _ { \widehat { \vec { P } _ { A - 1 } \vec { q } } } ~</td>
      <td>z _ { 1 } ^ { ( A ) } \simeq 1 - { \frac { E _ { m i n } } { M } } - \frac { | \vec { P } _ { A - 1 } | ^ { 2 } } { 2 M A _ { A - 1 } } + \frac { \eta } { M } | \vec { P } _ { A - 1 } | c o s \theta _ { \tilde { f } _ { A - \tilde { q } } \vec { q } }</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.076923</td>
      <td>65</td>
      <td>$\operatorname* { l i m } _ { b j } ( - p \cdot q ) ^ { 5 + l } \left( \xi ^ { 2 } W _ { 4 } \, ^ { + } - 2 \xi W _ { 5 } \, ^ { + } \right) = c ^ { 2 } \phi ( \xi ) \, . $</td>
      <td>67</td>
      <td>$\operatorname* { l i m } _ { g j } ( - p \cdot q ) ^ { 5 + l } \left( \xi ^ { 2 } W _ { 4 } { } ^ { + } - 2 \xi W _ { 5 } { } ^ { + } \right) = c ^ { 2 } \phi ( \xi ) \, . $</td>
      <td>\operatorname* { l i m } _ { b j } ( - p \cdot q ) ^ { 5 + l } \left( \xi ^ { 2 } W _ { 4 } \, ^ { + } - 2 \xi W _ { 5 } \, ^ { + } \right) = c ^ { 2 } \phi ( \xi ) \, .</td>
      <td>\operatorname* { l i m } _ { g j } ( - p \cdot q ) ^ { 5 + l } \left( \xi ^ { 2 } W _ { 4 } { } ^ { + } - 2 \xi W _ { 5 } { } ^ { + } \right) = c ^ { 2 } \phi ( \xi ) \, .</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.000000</td>
      <td>55</td>
      <td>$d _ { m } = { \frac { 1 } { r } } { \frac { \partial } { \partial r } } r { \frac { \partial } { \partial r } } - { \frac { m ^ { 2 } } { r ^ { 2 } } } . $</td>
      <td>55</td>
      <td>$d _ { m } = { \frac { 1 } { r } } { \frac { \partial } { \partial r } } r { \frac { \partial } { \partial r } } - { \frac { m ^ { 2 } } { r ^ { 2 } } } . $</td>
      <td>d _ { m } = { \frac { 1 } { r } } { \frac { \partial } { \partial r } } r { \frac { \partial } { \partial r } } - { \frac { m ^ { 2 } } { r ^ { 2 } } } .</td>
      <td>d _ { m } = { \frac { 1 } { r } } { \frac { \partial } { \partial r } } r { \frac { \partial } { \partial r } } - { \frac { m ^ { 2 } } { r ^ { 2 } } } .</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.000000</td>
      <td>49</td>
      <td>$\phi _ { 2 } ( z ) = \frac { a _ { 0 } + \cdots + a _ { N } z ^ { N } } { ( 1 + | z | ^ { 2 } ) ^ { N / 2 } } $</td>
      <td>49</td>
      <td>$\phi _ { 2 } ( z ) = \frac { a _ { 0 } + \cdots + a _ { N } z ^ { N } } { ( 1 + | z | ^ { 2 } ) ^ { N / 2 } } $</td>
      <td>\phi _ { 2 } ( z ) = \frac { a _ { 0 } + \cdots + a _ { N } z ^ { N } } { ( 1 + | z | ^ { 2 } ) ^ { N / 2 } }</td>
      <td>\phi _ { 2 } ( z ) = \frac { a _ { 0 } + \cdots + a _ { N } z ^ { N } } { ( 1 + | z | ^ { 2 } ) ^ { N / 2 } }</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.022222</td>
      <td>45</td>
      <td>$[ M _ { + - } , P _ { + } ] = i P _ { + } , \ [ M _ { + - } , P _ { - } ] = - i P _ { - } $</td>
      <td>45</td>
      <td>$[ M _ { + - } , P _ { + } ] = i P _ { + } , \; [ M _ { + - } , P _ { - } ] = - i P _ { - } $</td>
      <td>[ M _ { + - } , P _ { + } ] = i P _ { + } , \ [ M _ { + - } , P _ { - } ] = - i P _ { - }</td>
      <td>[ M _ { + - } , P _ { + } ] = i P _ { + } , \; [ M _ { + - } , P _ { - } ] = - i P _ { - }</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.000000</td>
      <td>27</td>
      <td>$\beta _ { n } ( \tau ) = b _ { n } e ^ { - i \omega _ { n } \tau } . $</td>
      <td>27</td>
      <td>$\beta _ { n } ( \tau ) = b _ { n } e ^ { - i \omega _ { n } \tau } . $</td>
      <td>\beta _ { n } ( \tau ) = b _ { n } e ^ { - i \omega _ { n } \tau } .</td>
      <td>\beta _ { n } ( \tau ) = b _ { n } e ^ { - i \omega _ { n } \tau } .</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.027397</td>
      <td>73</td>
      <td>$\sigma _ { \mathrm { R } } ^ { 2 } ( t ) = v _ { \mathrm { R } } ^ { 2 } ( t ) - \lambda _ { \mathrm { R } } \tilde { I } _ { 1 } ^ { \zeta } ( t ) + O ( \lambda _ { \mathrm { R } } ^ { 2 } ) \, , $</td>
      <td>75</td>
      <td>$\sigma _ { \mathrm { R } } ^ { 2 } ( t ) = v _ { \mathrm { R } } ^ { 2 } ( t ) - \lambda _ { \mathrm { R } } { \tilde { I } _ { 1 } ^ { \zeta } ( t ) } + O ( \lambda _ { \mathrm { R } } ^ { 2 } ) \, , $</td>
      <td>\sigma _ { \mathrm { R } } ^ { 2 } ( t ) = v _ { \mathrm { R } } ^ { 2 } ( t ) - \lambda _ { \mathrm { R } } \tilde { I } _ { 1 } ^ { \zeta } ( t ) + O ( \lambda _ { \mathrm { R } } ^ { 2 } ) \, ,</td>
      <td>\sigma _ { \mathrm { R } } ^ { 2 } ( t ) = v _ { \mathrm { R } } ^ { 2 } ( t ) - \lambda _ { \mathrm { R } } { \tilde { I } _ { 1 } ^ { \zeta } ( t ) } + O ( \lambda _ { \mathrm { R } } ^ { 2 } ) \, ,</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.000000</td>
      <td>44</td>
      <td>$f = i e ^ { m z + C _ { 2 } } \, , \qquad g = m z + C _ { 2 } - \operatorname { l o g } ( C _ { 1 } ) \, . $</td>
      <td>44</td>
      <td>$f = i e ^ { m z + C _ { 2 } } \, , \qquad g = m z + C _ { 2 } - \operatorname { l o g } ( C _ { 1 } ) \, . $</td>
      <td>f = i e ^ { m z + C _ { 2 } } \, , \qquad g = m z + C _ { 2 } - \operatorname { l o g } ( C _ { 1 } ) \, .</td>
      <td>f = i e ^ { m z + C _ { 2 } } \, , \qquad g = m z + C _ { 2 } - \operatorname { l o g } ( C _ { 1 } ) \, .</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.000000</td>
      <td>36</td>
      <td>$R ( x ) = e ^ { - k r _ { 0 } } \frac { r ( x ) } { \sqrt { 6 } M _ { P l } } . $</td>
      <td>36</td>
      <td>$R ( x ) = e ^ { - k r _ { 0 } } \frac { r ( x ) } { \sqrt { 6 } M _ { P l } } . $</td>
      <td>R ( x ) = e ^ { - k r _ { 0 } } \frac { r ( x ) } { \sqrt { 6 } M _ { P l } } .</td>
      <td>R ( x ) = e ^ { - k r _ { 0 } } \frac { r ( x ) } { \sqrt { 6 } M _ { P l } } .</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.203704</td>
      <td>54</td>
      <td>$Y _ { i j } ^ { \; \; \; k } = \sum _ { m } \frac { S _ { m i } P _ { m j } P _ { m } ^ { k } } { S _ { m 0 } } \; \; . $</td>
      <td>55</td>
      <td>${ Y _ { i j } } ^ { k } = \sum _ { m } { \frac { S _ { m i } P _ { n j } P _ { \eta } ^ { k } } { S _ { m 0 } } } \ \ . $</td>
      <td>Y _ { i j } ^ { \; \; \; k } = \sum _ { m } \frac { S _ { m i } P _ { m j } P _ { m } ^ { k } } { S _ { m 0 } } \; \; .</td>
      <td>{ Y _ { i j } } ^ { k } = \sum _ { m } { \frac { S _ { m i } P _ { n j } P _ { \eta } ^ { k } } { S _ { m 0 } } } \ \ .</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.019608</td>
      <td>51</td>
      <td>$\Gamma ( B \rightarrow X _ { s } \gamma ) _ { L O } ( \mu = \frac { m _ { b } } { 2 } ) \approx \Gamma ( B \rightarrow X _ { s } \gamma ) _ { N L O } \; . $</td>
      <td>50</td>
      <td>$\Gamma ( B \rightarrow X _ { s } \gamma ) _ { L O } ( \mu = \frac { m _ { b } } { 2 } ) \approx \Gamma ( B \rightarrow X _ { s } \gamma _ { N L O } \; . $</td>
      <td>\Gamma ( B \rightarrow X _ { s } \gamma ) _ { L O } ( \mu = \frac { m _ { b } } { 2 } ) \approx \Gamma ( B \rightarrow X _ { s } \gamma ) _ { N L O } \; .</td>
      <td>\Gamma ( B \rightarrow X _ { s } \gamma ) _ { L O } ( \mu = \frac { m _ { b } } { 2 } ) \approx \Gamma ( B \rightarrow X _ { s } \gamma _ { N L O } \; .</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.043478</td>
      <td>115</td>
      <td>$\langle 0 | ( X ^ { 1 } ( \sigma ) - \overline { { X ^ { 1 } } } ) ^ { 2 } | 0 \rangle \ = \ \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m ^ { 2 } } \langle 0 | ( \alpha _ { m } \alpha _ { - m } + \tilde { \alpha } _ { m } \tilde { \alpha } _ { - m } ) | 0 \rangle \ = 2 \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m } . $</td>
      <td>115</td>
      <td>$\langle 0 | ( X ^ { 1 } ( \sigma ) - \overline { { X ^ { 1 } } } ) ^ { 2 } | 0 \rangle \; = \; \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m ^ { 2 } } \langle 0 | ( \alpha _ { m } \alpha _ { - m } + \tilde { \alpha } _ { m } \tilde { a } _ { - m } ) | 0 \rangle \; = 2 \sum _ { m = 1 } ^ { \infty } \frac { \Gamma } { m } . $</td>
      <td>\langle 0 | ( X ^ { 1 } ( \sigma ) - \overline { { X ^ { 1 } } } ) ^ { 2 } | 0 \rangle \ = \ \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m ^ { 2 } } \langle 0 | ( \alpha _ { m } \alpha _ { - m } + \tilde { \alpha } _ { m } \tilde { \alpha } _ { - m } ) | 0 \rangle \ = 2 \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m } .</td>
      <td>\langle 0 | ( X ^ { 1 } ( \sigma ) - \overline { { X ^ { 1 } } } ) ^ { 2 } | 0 \rangle \; = \; \sum _ { m = 1 } ^ { \infty } \frac { 1 } { m ^ { 2 } } \langle 0 | ( \alpha _ { m } \alpha _ { - m } + \tilde { \alpha } _ { m } \tilde { a } _ { - m } ) | 0 \rangle \; = 2 \sum _ { m = 1 } ^ { \infty } \frac { \Gamma } { m } .</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.000000</td>
      <td>45</td>
      <td>$F _ { 1 } ^ { i } ( x , Q ^ { 2 } ) = \beta _ { i } ( Q ^ { 2 } ) x ^ { - \alpha _ { i } ( 0 ) } . $</td>
      <td>45</td>
      <td>$F _ { 1 } ^ { i } ( x , Q ^ { 2 } ) = \beta _ { i } ( Q ^ { 2 } ) x ^ { - \alpha _ { i } ( 0 ) } . $</td>
      <td>F _ { 1 } ^ { i } ( x , Q ^ { 2 } ) = \beta _ { i } ( Q ^ { 2 } ) x ^ { - \alpha _ { i } ( 0 ) } .</td>
      <td>F _ { 1 } ^ { i } ( x , Q ^ { 2 } ) = \beta _ { i } ( Q ^ { 2 } ) x ^ { - \alpha _ { i } ( 0 ) } .</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.056338</td>
      <td>71</td>
      <td>$d s ^ { 2 } = { \frac { r ^ { 2 } } { R ^ { 2 } } } \left( - f d t ^ { 2 } + d { \bf x } ^ { 2 } \right) + { \frac { R ^ { 2 } } { r ^ { 2 } f } } d r ^ { 2 } \, , $</td>
      <td>67</td>
      <td>$d s ^ { 2 } = \frac { r ^ { 2 } } { R ^ { 2 } } \left( - f d t ^ { 2 } + d { \bf x } ^ { 2 } \right) + \frac { R ^ { 2 } } { r ^ { 2 } f } d r ^ { 2 } \, , $</td>
      <td>d s ^ { 2 } = { \frac { r ^ { 2 } } { R ^ { 2 } } } \left( - f d t ^ { 2 } + d { \bf x } ^ { 2 } \right) + { \frac { R ^ { 2 } } { r ^ { 2 } f } } d r ^ { 2 } \, ,</td>
      <td>d s ^ { 2 } = \frac { r ^ { 2 } } { R ^ { 2 } } \left( - f d t ^ { 2 } + d { \bf x } ^ { 2 } \right) + \frac { R ^ { 2 } } { r ^ { 2 } f } d r ^ { 2 } \, ,</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.056338</td>
      <td>71</td>
      <td>$P _ { \mu \, i j } + Q _ { \mu \, i j } \equiv L _ { i } ^ { \alpha } ( \partial _ { \mu } \delta _ { \alpha } ^ { \, \beta } - g \, \epsilon _ { \alpha \beta \gamma } A _ { \mu } ^ { \gamma } ) L _ { \beta \; j } , $</td>
      <td>68</td>
      <td>$P _ { \mu i j } + Q _ { \mu i j } \equiv L _ { i } ^ { \alpha } ( \partial _ { \mu } \delta _ { \alpha } ^ { \beta } - g \, \epsilon _ { \alpha \beta \gamma } A _ { \mu } ^ { \gamma } ) L _ { \beta \, j } , $</td>
      <td>P _ { \mu \, i j } + Q _ { \mu \, i j } \equiv L _ { i } ^ { \alpha } ( \partial _ { \mu } \delta _ { \alpha } ^ { \, \beta } - g \, \epsilon _ { \alpha \beta \gamma } A _ { \mu } ^ { \gamma } ) L _ { \beta \; j } ,</td>
      <td>P _ { \mu i j } + Q _ { \mu i j } \equiv L _ { i } ^ { \alpha } ( \partial _ { \mu } \delta _ { \alpha } ^ { \beta } - g \, \epsilon _ { \alpha \beta \gamma } A _ { \mu } ^ { \gamma } ) L _ { \beta \, j } ,</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.034091</td>
      <td>88</td>
      <td>$\hat { \rho } _ { \mathrm { G } } = \left( \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { c o s } \varphi _ { \mathrm { G } } , \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { s i n } \varphi _ { \mathrm { G } } , \operatorname { c o s } \vartheta _ { \mathrm { G } } \right) \, . $</td>
      <td>88</td>
      <td>$\hat { \rho } _ { \mathrm { G } } = ( \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { c o s } \varphi _ { \mathrm { G } } , \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { s i n } \varphi _ { \mathrm { G } } , \operatorname { c o s } \vartheta _ { \mathrm { G } } ) ~ . $</td>
      <td>\hat { \rho } _ { \mathrm { G } } = \left( \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { c o s } \varphi _ { \mathrm { G } } , \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { s i n } \varphi _ { \mathrm { G } } , \operatorname { c o s } \vartheta _ { \mathrm { G } } \right) \, .</td>
      <td>\hat { \rho } _ { \mathrm { G } } = ( \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { c o s } \varphi _ { \mathrm { G } } , \operatorname { s i n } \vartheta _ { \mathrm { G } } \operatorname { s i n } \varphi _ { \mathrm { G } } , \operatorname { c o s } \vartheta _ { \mathrm { G } } ) ~ .</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.156250</td>
      <td>64</td>
      <td>$\left[ \gamma _ { n } , \gamma _ { m } \right] \ = \ \left[ \beta _ { n } , \beta _ { m } \right] \, = \, 0 \, , \quad \ \left[ \beta _ { n } , \gamma _ { m } \right] \, = \, \delta _ { n + m , 0 } \ . $</td>
      <td>63</td>
      <td>$[ \gamma _ { n } , \gamma _ { m } ] \; = \; [ \beta _ { n } , \beta _ { m } ] \, = \, 0 \, , \quad [ \beta _ { n } , \gamma _ { m } ] \, = \, \delta _ { n + m , 0 } \; . $</td>
      <td>\left[ \gamma _ { n } , \gamma _ { m } \right] \ = \ \left[ \beta _ { n } , \beta _ { m } \right] \, = \, 0 \, , \quad \ \left[ \beta _ { n } , \gamma _ { m } \right] \, = \, \delta _ { n + m , 0 } \ .</td>
      <td>[ \gamma _ { n } , \gamma _ { m } ] \; = \; [ \beta _ { n } , \beta _ { m } ] \, = \, 0 \, , \quad [ \beta _ { n } , \gamma _ { m } ] \, = \, \delta _ { n + m , 0 } \; .</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.000000</td>
      <td>51</td>
      <td>$\widetilde { F } _ { a b } = \widetilde { F } _ { a b } ^ { r } = \widetilde { H } _ { a b c } = \sigma = \chi _ { A } = \psi _ { A \mu } = 0 $</td>
      <td>51</td>
      <td>$\widetilde { F } _ { a b } = \widetilde { F } _ { a b } ^ { r } = \widetilde { H } _ { a b c } = \sigma = \chi _ { A } = \psi _ { A \mu } = 0 $</td>
      <td>\widetilde { F } _ { a b } = \widetilde { F } _ { a b } ^ { r } = \widetilde { H } _ { a b c } = \sigma = \chi _ { A } = \psi _ { A \mu } = 0</td>
      <td>\widetilde { F } _ { a b } = \widetilde { F } _ { a b } ^ { r } = \widetilde { H } _ { a b c } = \sigma = \chi _ { A } = \psi _ { A \mu } = 0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.100000</td>
      <td>100</td>
      <td>$\Omega = \frac { 2 i } { \pi } \left( \frac { \delta m ^ { 2 } } { 4 E \hbar } \right) \int _ { t _ { 0 } } ^ { t _ { 0 } ^ { * } } { d t \sqrt { \operatorname { s i n } ^ { 2 } { 2 \theta _ { v } } + ( \zeta ( t ) - \operatorname { c o s } { 2 \theta _ { v } } ) ^ { 2 } } } \, . $</td>
      <td>90</td>
      <td>$\Omega = \frac { 2 i } { \pi } \left( \frac { \delta m ^ { 2 } } { 4 E \hbar } \right) \int _ { t _ { 0 } } ^ { t _ { 0 } } d t \sqrt { \operatorname { s i n } ^ { 2 } 2 \theta _ { v } + ( \zeta ( t ) - \operatorname { c o s } 2 \theta _ { v } ) ^ { 2 } } \, . $</td>
      <td>\Omega = \frac { 2 i } { \pi } \left( \frac { \delta m ^ { 2 } } { 4 E \hbar } \right) \int _ { t _ { 0 } } ^ { t _ { 0 } ^ { * } } { d t \sqrt { \operatorname { s i n } ^ { 2 } { 2 \theta _ { v } } + ( \zeta ( t ) - \operatorname { c o s } { 2 \theta _ { v } } ) ^ { 2 } } } \, .</td>
      <td>\Omega = \frac { 2 i } { \pi } \left( \frac { \delta m ^ { 2 } } { 4 E \hbar } \right) \int _ { t _ { 0 } } ^ { t _ { 0 } } d t \sqrt { \operatorname { s i n } ^ { 2 } 2 \theta _ { v } + ( \zeta ( t ) - \operatorname { c o s } 2 \theta _ { v } ) ^ { 2 } } \, .</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.000000</td>
      <td>68</td>
      <td>$2 \kappa N = 1 + N ^ { 2 } + 2 U ^ { 2 } + V ^ { 2 } - 2 V _ { 2 } + 2 \alpha ^ { 2 } W ^ { 2 } B ^ { 2 } - \alpha ^ { 2 } ( C + B ( \kappa - N ) ) ^ { 2 } $</td>
      <td>68</td>
      <td>$2 \kappa N = 1 + N ^ { 2 } + 2 U ^ { 2 } + V ^ { 2 } - 2 V _ { 2 } + 2 \alpha ^ { 2 } W ^ { 2 } B ^ { 2 } - \alpha ^ { 2 } ( C + B ( \kappa - N ) ) ^ { 2 } $</td>
      <td>2 \kappa N = 1 + N ^ { 2 } + 2 U ^ { 2 } + V ^ { 2 } - 2 V _ { 2 } + 2 \alpha ^ { 2 } W ^ { 2 } B ^ { 2 } - \alpha ^ { 2 } ( C + B ( \kappa - N ) ) ^ { 2 }</td>
      <td>2 \kappa N = 1 + N ^ { 2 } + 2 U ^ { 2 } + V ^ { 2 } - 2 V _ { 2 } + 2 \alpha ^ { 2 } W ^ { 2 } B ^ { 2 } - \alpha ^ { 2 } ( C + B ( \kappa - N ) ) ^ { 2 }</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.069767</td>
      <td>43</td>
      <td>$\rho _ { 0 } \propto \left| { \frac { G ^ { \prime \prime } ( x = 0 ) } { G ( x = 0 ) } } \right| ^ { \frac { N } { 2 } } . $</td>
      <td>42</td>
      <td>$\rho _ { 0 } \propto \left| \frac { G ^ { \prime \prime } ( x = 0 ) } { G ( x = 0 ) } \right| ^ { \frac { N } { 2 } } \, . $</td>
      <td>\rho _ { 0 } \propto \left| { \frac { G ^ { \prime \prime } ( x = 0 ) } { G ( x = 0 ) } } \right| ^ { \frac { N } { 2 } } .</td>
      <td>\rho _ { 0 } \propto \left| \frac { G ^ { \prime \prime } ( x = 0 ) } { G ( x = 0 ) } \right| ^ { \frac { N } { 2 } } \, .</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.098361</td>
      <td>122</td>
      <td>$\mathrm { D } _ { z } ^ { ( s ) } ( E _ { i } ) = z ^ { s _ { i } } E _ { i } \, , ~ ~ ~ \mathrm { D } _ { z } ^ { ( s ) } ( F _ { i } ) = z ^ { - s _ { i } } F _ { i } \, , ~ ~ ~ \mathrm { D } _ { z } ^ { ( s ) } ( h _ { i } ) = h _ { i } \, , ~ ~ ~ i = 0 , \cdots , r $</td>
      <td>113</td>
      <td>$\mathrm { D } _ { z } ^ { ( s ) } ( E _ { i } ) = z ^ { s _ { i } } E _ { i } \, , \quad \mathrm { D } _ { z } ^ { ( s ) } ( F _ { i } ) = z ^ { - s _ { i } } F _ { i } \, , \quad D _ { z } ^ { ( s ) } ( h _ { i } ) = h _ { i } \, , \quad i = 0 , \cdots , r $</td>
      <td>\mathrm { D } _ { z } ^ { ( s ) } ( E _ { i } ) = z ^ { s _ { i } } E _ { i } \, , ~ ~ ~ \mathrm { D } _ { z } ^ { ( s ) } ( F _ { i } ) = z ^ { - s _ { i } } F _ { i } \, , ~ ~ ~ \mathrm { D } _ { z } ^ { ( s ) } ( h _ { i } ) = h _ { i } \, , ~ ~ ~ i = 0 , \cdots , r</td>
      <td>\mathrm { D } _ { z } ^ { ( s ) } ( E _ { i } ) = z ^ { s _ { i } } E _ { i } \, , \quad \mathrm { D } _ { z } ^ { ( s ) } ( F _ { i } ) = z ^ { - s _ { i } } F _ { i } \, , \quad D _ { z } ^ { ( s ) } ( h _ { i } ) = h _ { i } \, , \quad i = 0 , \cdots , r</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.000000</td>
      <td>21</td>
      <td>$\Gamma ( b , s ) = 1 - e ^ { i \chi ( b , s ) } . $</td>
      <td>21</td>
      <td>$\Gamma ( b , s ) = 1 - e ^ { i \chi ( b , s ) } . $</td>
      <td>\Gamma ( b , s ) = 1 - e ^ { i \chi ( b , s ) } .</td>
      <td>\Gamma ( b , s ) = 1 - e ^ { i \chi ( b , s ) } .</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.000000</td>
      <td>30</td>
      <td>$1 - \sqrt { 1 - ( \beta ^ { 2 } - x ) } \simeq ( \beta ^ { 2 } - x ) / 2 \ , $</td>
      <td>30</td>
      <td>$1 - \sqrt { 1 - ( \beta ^ { 2 } - x ) } \simeq ( \beta ^ { 2 } - x ) / 2 \ , $</td>
      <td>1 - \sqrt { 1 - ( \beta ^ { 2 } - x ) } \simeq ( \beta ^ { 2 } - x ) / 2 \ ,</td>
      <td>1 - \sqrt { 1 - ( \beta ^ { 2 } - x ) } \simeq ( \beta ^ { 2 } - x ) / 2 \ ,</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.000000</td>
      <td>33</td>
      <td>$U ^ { \prime } ( \sigma _ { 0 } ) = 0 \mathrm { ~ a n d ~ } U ( \sigma _ { 0 } ) = 0 , $</td>
      <td>33</td>
      <td>$U ^ { \prime } ( \sigma _ { 0 } ) = 0 \mathrm { ~ a n d ~ } U ( \sigma _ { 0 } ) = 0 , $</td>
      <td>U ^ { \prime } ( \sigma _ { 0 } ) = 0 \mathrm { ~ a n d ~ } U ( \sigma _ { 0 } ) = 0 ,</td>
      <td>U ^ { \prime } ( \sigma _ { 0 } ) = 0 \mathrm { ~ a n d ~ } U ( \sigma _ { 0 } ) = 0 ,</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0.131313</td>
      <td>99</td>
      <td>$b _ { k } ^ { \mathrm { s } } = \epsilon _ { k i j } \partial _ { i } a _ { j } ^ { \mathrm { s } } = \epsilon _ { k i j } \partial _ { i } \left( \frac { 1 } { 2 i g } \mathrm { t r } \, \tau _ { 3 } \, \Omega _ { \mathrm { D } } \partial _ { j } \Omega _ { \mathrm { D } } ^ { \dagger } \right) , $</td>
      <td>87</td>
      <td>$b _ { k } ^ { s } = \epsilon _ { k i j } \partial _ { i } a _ { j } ^ { s } = \epsilon _ { k i j } \partial _ { i } \left( \frac { 1 } { 2 i g } \mathrm { t r } \, \tau _ { 3 } \, \Omega _ { D } \partial _ { j } \Omega _ { 0 } ^ { \dagger } \right) , $</td>
      <td>b _ { k } ^ { \mathrm { s } } = \epsilon _ { k i j } \partial _ { i } a _ { j } ^ { \mathrm { s } } = \epsilon _ { k i j } \partial _ { i } \left( \frac { 1 } { 2 i g } \mathrm { t r } \, \tau _ { 3 } \, \Omega _ { \mathrm { D } } \partial _ { j } \Omega _ { \mathrm { D } } ^ { \dagger } \right) ,</td>
      <td>b _ { k } ^ { s } = \epsilon _ { k i j } \partial _ { i } a _ { j } ^ { s } = \epsilon _ { k i j } \partial _ { i } \left( \frac { 1 } { 2 i g } \mathrm { t r } \, \tau _ { 3 } \, \Omega _ { D } \partial _ { j } \Omega _ { 0 } ^ { \dagger } \right) ,</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0.000000</td>
      <td>39</td>
      <td>$R _ { | n \rangle } ^ { ( m ) } ( u ) = b _ { n } ^ { m } ( u ) R ^ { ( m ) } ( u ) $</td>
      <td>39</td>
      <td>$R _ { | n \rangle } ^ { ( m ) } ( u ) = b _ { n } ^ { m } ( u ) R ^ { ( m ) } ( u ) $</td>
      <td>R _ { | n \rangle } ^ { ( m ) } ( u ) = b _ { n } ^ { m } ( u ) R ^ { ( m ) } ( u )</td>
      <td>R _ { | n \rangle } ^ { ( m ) } ( u ) = b _ { n } ^ { m } ( u ) R ^ { ( m ) } ( u )</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.000000</td>
      <td>33</td>
      <td>$\sigma _ { x x } ^ { 2 } + ( \sigma _ { x y } - 1 / 2 ) ^ { 2 } = 1 / 4 \; , $</td>
      <td>33</td>
      <td>$\sigma _ { x x } ^ { 2 } + ( \sigma _ { x y } - 1 / 2 ) ^ { 2 } = 1 / 4 \; , $</td>
      <td>\sigma _ { x x } ^ { 2 } + ( \sigma _ { x y } - 1 / 2 ) ^ { 2 } = 1 / 4 \; ,</td>
      <td>\sigma _ { x x } ^ { 2 } + ( \sigma _ { x y } - 1 / 2 ) ^ { 2 } = 1 / 4 \; ,</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.034722</td>
      <td>144</td>
      <td>$\int d ^ { n } p \, d ^ { n } q \, \frac { p ^ { \mu _ { 1 } } \ldots p ^ { \mu _ { i } } q ^ { \mu _ { i + 1 } } \ldots q ^ { \mu _ { j } } } { ( p ^ { 2 } + m _ { 1 } ^ { 2 } ) ^ { \alpha _ { 1 } } \, ( q ^ { 2 } + m _ { 2 } ^ { 2 } ) ^ { \alpha _ { 2 } } \, [ ( r + k ) ^ { 2 } + m _ { 3 } ^ { 2 } ] ^ { \alpha _ { 3 } } } \; \; . $</td>
      <td>145</td>
      <td>$\int d ^ { n } p \, d ^ { n } q \, \frac { p ^ { \mu _ { 1 } } \cdots p ^ { \mu _ { i } } q ^ { \mu _ { i + 4 } } \ldots q ^ { \mu _ { j } } } { ( p ^ { 2 } + m _ { 1 } ^ { 2 } ) ^ { \alpha _ { 1 } } \, ( q ^ { 2 } + m _ { 2 } ^ { 2 } ) ^ { \alpha _ { 2 } } \, [ ( r + k ) ^ { 2 } + m _ { 3 } ^ { 2 } ] ^ { \alpha _ { 3 } } } \, \, \, . $</td>
      <td>\int d ^ { n } p \, d ^ { n } q \, \frac { p ^ { \mu _ { 1 } } \ldots p ^ { \mu _ { i } } q ^ { \mu _ { i + 1 } } \ldots q ^ { \mu _ { j } } } { ( p ^ { 2 } + m _ { 1 } ^ { 2 } ) ^ { \alpha _ { 1 } } \, ( q ^ { 2 } + m _ { 2 } ^ { 2 } ) ^ { \alpha _ { 2 } } \, [ ( r + k ) ^ { 2 } + m _ { 3 } ^ { 2 } ] ^ { \alpha _ { 3 } } } \; \; .</td>
      <td>\int d ^ { n } p \, d ^ { n } q \, \frac { p ^ { \mu _ { 1 } } \cdots p ^ { \mu _ { i } } q ^ { \mu _ { i + 4 } } \ldots q ^ { \mu _ { j } } } { ( p ^ { 2 } + m _ { 1 } ^ { 2 } ) ^ { \alpha _ { 1 } } \, ( q ^ { 2 } + m _ { 2 } ^ { 2 } ) ^ { \alpha _ { 2 } } \, [ ( r + k ) ^ { 2 } + m _ { 3 } ^ { 2 } ] ^ { \alpha _ { 3 } } } \, \, \, .</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.066667</td>
      <td>60</td>
      <td>$S _ { s y m } ^ { q , 1 } = 2 w \int _ { y = y _ { - } } ^ { y = y _ { c } } \sqrt { 4 m _ { q } V _ { q p a i r } } ~ d y \quad , $</td>
      <td>57</td>
      <td>$S _ { s y m } ^ { q , 1 } = 2 w \int _ { y = y - } ^ { y = y _ { c } } \sqrt { 4 m _ { q } V _ { q p a i r } } \; d y \quad , $</td>
      <td>S _ { s y m } ^ { q , 1 } = 2 w \int _ { y = y _ { - } } ^ { y = y _ { c } } \sqrt { 4 m _ { q } V _ { q p a i r } } ~ d y \quad ,</td>
      <td>S _ { s y m } ^ { q , 1 } = 2 w \int _ { y = y - } ^ { y = y _ { c } } \sqrt { 4 m _ { q } V _ { q p a i r } } \; d y \quad ,</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.000000</td>
      <td>23</td>
      <td>${ \frac { n _ { M } } { s } } \leq 1 0 ^ { - 3 1 } . $</td>
      <td>23</td>
      <td>${ \frac { n _ { M } } { s } } \leq 1 0 ^ { - 3 1 } . $</td>
      <td>{ \frac { n _ { M } } { s } } \leq 1 0 ^ { - 3 1 } .</td>
      <td>{ \frac { n _ { M } } { s } } \leq 1 0 ^ { - 3 1 } .</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.000000</td>
      <td>41</td>
      <td>$\alpha = - 4 ( 8 e _ { 3 8 } + e _ { 1 1 5 } + e _ { 1 1 6 } ) , \quad \beta = 4 e _ { 2 2 } . $</td>
      <td>41</td>
      <td>$\alpha = - 4 ( 8 e _ { 3 8 } + e _ { 1 1 5 } + e _ { 1 1 6 } ) , \quad \beta = 4 e _ { 2 2 } . $</td>
      <td>\alpha = - 4 ( 8 e _ { 3 8 } + e _ { 1 1 5 } + e _ { 1 1 6 } ) , \quad \beta = 4 e _ { 2 2 } .</td>
      <td>\alpha = - 4 ( 8 e _ { 3 8 } + e _ { 1 1 5 } + e _ { 1 1 6 } ) , \quad \beta = 4 e _ { 2 2 } .</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.180000</td>
      <td>50</td>
      <td>$\operatorname* { l i m } _ { \beta \rightarrow \infty } { } _ { 1 } F _ { 1 } ( \beta ; \gamma ; z / \beta ) = { } _ { 0 } F _ { 1 } ( \gamma ; z ) , $</td>
      <td>42</td>
      <td>$\operatorname* { l i m } _ { \beta \rightarrow \infty } \Gamma _ { 1 } ( \beta ; \gamma ; z / \beta ) = _ { 0 } F _ { 1 } ( \gamma ; z ) , $</td>
      <td>\operatorname* { l i m } _ { \beta \rightarrow \infty } { } _ { 1 } F _ { 1 } ( \beta ; \gamma ; z / \beta ) = { } _ { 0 } F _ { 1 } ( \gamma ; z ) ,</td>
      <td>\operatorname* { l i m } _ { \beta \rightarrow \infty } \Gamma _ { 1 } ( \beta ; \gamma ; z / \beta ) = _ { 0 } F _ { 1 } ( \gamma ; z ) ,</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.000000</td>
      <td>14</td>
      <td>$j ^ { \nu } \, w _ { \nu \sigma } = 0 $</td>
      <td>14</td>
      <td>$j ^ { \nu } \, w _ { \nu \sigma } = 0 $</td>
      <td>j ^ { \nu } \, w _ { \nu \sigma } = 0</td>
      <td>j ^ { \nu } \, w _ { \nu \sigma } = 0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.000000</td>
      <td>30</td>
      <td>$\Psi _ { H } ^ { \pi } = V ^ { \alpha _ { s } } \otimes \Psi _ { S } ^ { \pi } . $</td>
      <td>30</td>
      <td>$\Psi _ { H } ^ { \pi } = V ^ { \alpha _ { s } } \otimes \Psi _ { S } ^ { \pi } . $</td>
      <td>\Psi _ { H } ^ { \pi } = V ^ { \alpha _ { s } } \otimes \Psi _ { S } ^ { \pi } .</td>
      <td>\Psi _ { H } ^ { \pi } = V ^ { \alpha _ { s } } \otimes \Psi _ { S } ^ { \pi } .</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.000000</td>
      <td>58</td>
      <td>$G _ { A B } ^ { ( 5 ) } = - \Lambda _ { ( 5 ) } g _ { A B } ^ { ( 5 ) } + \kappa _ { ( 5 ) } ^ { 2 } T _ { A B } ^ { ( 5 ) } , $</td>
      <td>58</td>
      <td>$G _ { A B } ^ { ( 5 ) } = - \Lambda _ { ( 5 ) } g _ { A B } ^ { ( 5 ) } + \kappa _ { ( 5 ) } ^ { 2 } T _ { A B } ^ { ( 5 ) } , $</td>
      <td>G _ { A B } ^ { ( 5 ) } = - \Lambda _ { ( 5 ) } g _ { A B } ^ { ( 5 ) } + \kappa _ { ( 5 ) } ^ { 2 } T _ { A B } ^ { ( 5 ) } ,</td>
      <td>G _ { A B } ^ { ( 5 ) } = - \Lambda _ { ( 5 ) } g _ { A B } ^ { ( 5 ) } + \kappa _ { ( 5 ) } ^ { 2 } T _ { A B } ^ { ( 5 ) } ,</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.021739</td>
      <td>92</td>
      <td>$V _ { e f f } ( b o s o n ) = \frac { 1 } { 2 } t r \ \int \frac { d \omega } { 2 \pi } \int _ { 0 } ^ { \infty } \frac { d s } { s } e ^ { i \omega \tau } e x p [ - s ( - \partial _ { \tau } ^ { 2 } + W ( \tau ) ) ] e ^ { - i \omega \tau } \, $</td>
      <td>91</td>
      <td>$V _ { e f f } ( b o s o n ) = \frac { 1 } { 2 } t r \; \int \frac { d \omega } { 2 \pi } \int _ { 0 } ^ { \infty } \frac { d s } { s } e ^ { i \omega \tau } e x p [ - s ( - \partial _ { \tau } ^ { 2 } + W ( \tau ) ) ] e ^ { - i \omega \tau } $</td>
      <td>V _ { e f f } ( b o s o n ) = \frac { 1 } { 2 } t r \ \int \frac { d \omega } { 2 \pi } \int _ { 0 } ^ { \infty } \frac { d s } { s } e ^ { i \omega \tau } e x p [ - s ( - \partial _ { \tau } ^ { 2 } + W ( \tau ) ) ] e ^ { - i \omega \tau } \,</td>
      <td>V _ { e f f } ( b o s o n ) = \frac { 1 } { 2 } t r \; \int \frac { d \omega } { 2 \pi } \int _ { 0 } ^ { \infty } \frac { d s } { s } e ^ { i \omega \tau } e x p [ - s ( - \partial _ { \tau } ^ { 2 } + W ( \tau ) ) ] e ^ { - i \omega \tau }</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.000000</td>
      <td>50</td>
      <td>$\epsilon = \frac { 1 } { \sqrt { 2 } } ( \frac { I m M _ { 1 2 } } { 2 R e M _ { 1 2 } } + \xi _ { 0 } ) e ^ { i \pi / 4 } $</td>
      <td>50</td>
      <td>$\epsilon = \frac { 1 } { \sqrt { 2 } } ( \frac { I m M _ { 1 2 } } { 2 R e M _ { 1 2 } } + \xi _ { 0 } ) e ^ { i \pi / 4 } $</td>
      <td>\epsilon = \frac { 1 } { \sqrt { 2 } } ( \frac { I m M _ { 1 2 } } { 2 R e M _ { 1 2 } } + \xi _ { 0 } ) e ^ { i \pi / 4 }</td>
      <td>\epsilon = \frac { 1 } { \sqrt { 2 } } ( \frac { I m M _ { 1 2 } } { 2 R e M _ { 1 2 } } + \xi _ { 0 } ) e ^ { i \pi / 4 }</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.181818</td>
      <td>66</td>
      <td>$d X ^ { \underline { { m } } } = { \frac { 1 } { 2 } } e ^ { + + } u ^ { -- \underline { { m } } } + { \frac { 1 } { 2 } } e ^ { -- } u ^ { + + \underline { { m } } } , $</td>
      <td>61</td>
      <td>$d X ^ { \underline { m } } = { \frac { 1 } { 2 } } e ^ { + + } u ^ { -- } \longrightarrow _ { \underline { m } } + { \frac { 1 } { 2 } } e ^ { -- } u ^ { + + m } , $</td>
      <td>d X ^ { \underline { { m } } } = { \frac { 1 } { 2 } } e ^ { + + } u ^ { -- \underline { { m } } } + { \frac { 1 } { 2 } } e ^ { -- } u ^ { + + \underline { { m } } } ,</td>
      <td>d X ^ { \underline { m } } = { \frac { 1 } { 2 } } e ^ { + + } u ^ { -- } \longrightarrow _ { \underline { m } } + { \frac { 1 } { 2 } } e ^ { -- } u ^ { + + m } ,</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.035088</td>
      <td>57</td>
      <td>$V ( y ) = { \frac { a ^ { 2 } c _ { 2 } - b ^ { 2 } c _ { 1 } \operatorname { s i n h } ( a y ) } { a b \operatorname { c o s h } ( a y ) } } $</td>
      <td>55</td>
      <td>$V ( y ) = \frac { a ^ { 2 } c _ { 2 } - b ^ { 2 } c _ { 1 } \operatorname { s i n h } ( a y ) } { a b \operatorname { c o s h } ( a y ) } $</td>
      <td>V ( y ) = { \frac { a ^ { 2 } c _ { 2 } - b ^ { 2 } c _ { 1 } \operatorname { s i n h } ( a y ) } { a b \operatorname { c o s h } ( a y ) } }</td>
      <td>V ( y ) = \frac { a ^ { 2 } c _ { 2 } - b ^ { 2 } c _ { 1 } \operatorname { s i n h } ( a y ) } { a b \operatorname { c o s h } ( a y ) }</td>
    </tr>
    <tr>
      <th>66</th>
      <td>0.093750</td>
      <td>64</td>
      <td>$m _ { \lambda } \simeq { \frac { \langle F _ { X _ { 1 } } \rangle } { M ^ { 3 } } } \simeq \lambda { \frac { v ^ { 8 } } { M ^ { 7 } } } \sim 1 0 ^ { - 4 } \, \mathrm { G e V } . $</td>
      <td>64</td>
      <td>$m _ { \lambda } \simeq \frac { \langle F _ { X _ { 1 } } \rangle } { M ^ { 3 } } \simeq \lambda ^ { 2 } \frac { v ^ { 8 } } { M ^ { 7 } } \sim 1 0 ^ { - 4 } \, \mathrm { G e V } . $</td>
      <td>m _ { \lambda } \simeq { \frac { \langle F _ { X _ { 1 } } \rangle } { M ^ { 3 } } } \simeq \lambda { \frac { v ^ { 8 } } { M ^ { 7 } } } \sim 1 0 ^ { - 4 } \, \mathrm { G e V } .</td>
      <td>m _ { \lambda } \simeq \frac { \langle F _ { X _ { 1 } } \rangle } { M ^ { 3 } } \simeq \lambda ^ { 2 } \frac { v ^ { 8 } } { M ^ { 7 } } \sim 1 0 ^ { - 4 } \, \mathrm { G e V } .</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.030769</td>
      <td>65</td>
      <td>$\bar { R } ^ { 2 } ( m , R ) \, = \, R _ { c } ^ { 2 } \, \left[ 1 \, - \, \frac { 2 4 } { \pi } \, I \left( \frac { M _ { 0 } ^ { 2 } \, R } { m } \right) \right] \, { . } $</td>
      <td>63</td>
      <td>$\bar { R } ^ { 2 } ( m , R ) \, = \, R _ { c } ^ { 2 } \, \left[ 1 \, - \, \frac { 2 4 } { \pi } \, I \left( \frac { M _ { 0 } ^ { 2 } \, R } { m } \right) \right] \, . $</td>
      <td>\bar { R } ^ { 2 } ( m , R ) \, = \, R _ { c } ^ { 2 } \, \left[ 1 \, - \, \frac { 2 4 } { \pi } \, I \left( \frac { M _ { 0 } ^ { 2 } \, R } { m } \right) \right] \, { . }</td>
      <td>\bar { R } ^ { 2 } ( m , R ) \, = \, R _ { c } ^ { 2 } \, \left[ 1 \, - \, \frac { 2 4 } { \pi } \, I \left( \frac { M _ { 0 } ^ { 2 } \, R } { m } \right) \right] \, .</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.044944</td>
      <td>89</td>
      <td>$L = { \frac { 1 } { 2 } } \bigl ( \dot { r } ^ { 2 } + r ^ { 2 } ( \dot { \theta } ^ { 2 } + \operatorname { s i n } ^ { 2 } \theta \dot { \phi } ^ { 2 } ) \bigr ) - \lambda ( r ^ { 2 } - a ^ { 2 } ) ^ { 2 } - \Delta r \operatorname { c o s } \theta . $</td>
      <td>85</td>
      <td>$L = \frac { 1 } { 2 } ( \dot { r } ^ { 2 } + r ^ { 2 } ( \dot { \theta } ^ { 2 } + \operatorname { s i n } ^ { 2 } \theta \dot { \phi } ^ { 2 } ) ) - \lambda ( r ^ { 2 } - a ^ { 2 } ) ^ { 2 } - \Delta r \operatorname { c o s } \theta . $</td>
      <td>L = { \frac { 1 } { 2 } } \bigl ( \dot { r } ^ { 2 } + r ^ { 2 } ( \dot { \theta } ^ { 2 } + \operatorname { s i n } ^ { 2 } \theta \dot { \phi } ^ { 2 } ) \bigr ) - \lambda ( r ^ { 2 } - a ^ { 2 } ) ^ { 2 } - \Delta r \operatorname { c o s } \theta .</td>
      <td>L = \frac { 1 } { 2 } ( \dot { r } ^ { 2 } + r ^ { 2 } ( \dot { \theta } ^ { 2 } + \operatorname { s i n } ^ { 2 } \theta \dot { \phi } ^ { 2 } ) ) - \lambda ( r ^ { 2 } - a ^ { 2 } ) ^ { 2 } - \Delta r \operatorname { c o s } \theta .</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.187500</td>
      <td>16</td>
      <td>$\frac { \partial V ( \star \phi ) } { \partial \phi } = 0 . $</td>
      <td>18</td>
      <td>${ \frac { \partial V ( * \phi ) } { \partial \phi } } = 0 . $</td>
      <td>\frac { \partial V ( \star \phi ) } { \partial \phi } = 0 .</td>
      <td>{ \frac { \partial V ( * \phi ) } { \partial \phi } } = 0 .</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.051724</td>
      <td>58</td>
      <td>$\langle \Gamma _ { 1 } , \widehat { \Gamma _ { 2 } } \rangle = \langle \Gamma _ { 1 } , * _ { X } \Gamma _ { 2 } \rangle = \int _ { X } \Gamma _ { 1 } \wedge * _ { X } \Gamma _ { 2 } . $</td>
      <td>58</td>
      <td>$\langle \Gamma _ { 1 } , \widetilde { \Gamma } _ { 2 } \rangle = \langle \Gamma _ { 1 } , * _ { X } \Gamma _ { 2 } \rangle = \int _ { X } \Gamma _ { 1 } \wedge * _ { X } \Gamma _ { 2 } . $</td>
      <td>\langle \Gamma _ { 1 } , \widehat { \Gamma _ { 2 } } \rangle = \langle \Gamma _ { 1 } , * _ { X } \Gamma _ { 2 } \rangle = \int _ { X } \Gamma _ { 1 } \wedge * _ { X } \Gamma _ { 2 } .</td>
      <td>\langle \Gamma _ { 1 } , \widetilde { \Gamma } _ { 2 } \rangle = \langle \Gamma _ { 1 } , * _ { X } \Gamma _ { 2 } \rangle = \int _ { X } \Gamma _ { 1 } \wedge * _ { X } \Gamma _ { 2 } .</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.000000</td>
      <td>47</td>
      <td>$Z = 1 + \lambda ^ { 2 } \frac { \hbar } { 3 2 \pi ^ { 2 } } ( 1 + \operatorname { l o g } \frac { M ^ { 2 } } { \mu ^ { 2 } } ) $</td>
      <td>47</td>
      <td>$Z = 1 + \lambda ^ { 2 } \frac { \hbar } { 3 2 \pi ^ { 2 } } ( 1 + \operatorname { l o g } \frac { M ^ { 2 } } { \mu ^ { 2 } } ) $</td>
      <td>Z = 1 + \lambda ^ { 2 } \frac { \hbar } { 3 2 \pi ^ { 2 } } ( 1 + \operatorname { l o g } \frac { M ^ { 2 } } { \mu ^ { 2 } } )</td>
      <td>Z = 1 + \lambda ^ { 2 } \frac { \hbar } { 3 2 \pi ^ { 2 } } ( 1 + \operatorname { l o g } \frac { M ^ { 2 } } { \mu ^ { 2 } } )</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.000000</td>
      <td>76</td>
      <td>$\tilde { \chi } _ { 1 } ^ { 0 } = N _ { 1 1 } \widetilde { B } + N _ { 1 2 } \widetilde { W } ^ { 3 } + N _ { 1 3 } \widetilde { H } _ { 1 } ^ { 0 } + N _ { 1 4 } \widetilde { H } _ { 2 } ^ { 0 } $</td>
      <td>76</td>
      <td>$\tilde { \chi } _ { 1 } ^ { 0 } = N _ { 1 1 } \widetilde { B } + N _ { 1 2 } \widetilde { W } ^ { 3 } + N _ { 1 3 } \widetilde { H } _ { 1 } ^ { 0 } + N _ { 1 4 } \widetilde { H } _ { 2 } ^ { 0 } $</td>
      <td>\tilde { \chi } _ { 1 } ^ { 0 } = N _ { 1 1 } \widetilde { B } + N _ { 1 2 } \widetilde { W } ^ { 3 } + N _ { 1 3 } \widetilde { H } _ { 1 } ^ { 0 } + N _ { 1 4 } \widetilde { H } _ { 2 } ^ { 0 }</td>
      <td>\tilde { \chi } _ { 1 } ^ { 0 } = N _ { 1 1 } \widetilde { B } + N _ { 1 2 } \widetilde { W } ^ { 3 } + N _ { 1 3 } \widetilde { H } _ { 1 } ^ { 0 } + N _ { 1 4 } \widetilde { H } _ { 2 } ^ { 0 }</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.000000</td>
      <td>83</td>
      <td>$( M _ { W } ^ { 0 } ) ^ { 2 } = M _ { W } ^ { 2 } \ + \ \delta M _ { W } ^ { 2 } \, , \qquad ( M _ { Z } ^ { 0 } ) ^ { 2 } \ = \ M _ { Z } ^ { 2 } \ + \ \delta M _ { Z } ^ { 2 } \, , $</td>
      <td>83</td>
      <td>$( M _ { W } ^ { 0 } ) ^ { 2 } = M _ { W } ^ { 2 } \ + \ \delta M _ { W } ^ { 2 } \, , \qquad ( M _ { Z } ^ { 0 } ) ^ { 2 } \ = \ M _ { Z } ^ { 2 } \ + \ \delta M _ { Z } ^ { 2 } \, , $</td>
      <td>( M _ { W } ^ { 0 } ) ^ { 2 } = M _ { W } ^ { 2 } \ + \ \delta M _ { W } ^ { 2 } \, , \qquad ( M _ { Z } ^ { 0 } ) ^ { 2 } \ = \ M _ { Z } ^ { 2 } \ + \ \delta M _ { Z } ^ { 2 } \, ,</td>
      <td>( M _ { W } ^ { 0 } ) ^ { 2 } = M _ { W } ^ { 2 } \ + \ \delta M _ { W } ^ { 2 } \, , \qquad ( M _ { Z } ^ { 0 } ) ^ { 2 } \ = \ M _ { Z } ^ { 2 } \ + \ \delta M _ { Z } ^ { 2 } \, ,</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.062937</td>
      <td>143</td>
      <td>$\tau _ { 0 } ( y ) = \sum _ { \sigma _ { 1 } = 0 } ^ { 1 } . . \sum _ { \sigma _ { 4 } = 0 } ^ { 1 } Y _ { \sigma _ { 1 } . . \sigma _ { 4 } } ^ { ( 0 ) } ( t _ { 1 } Z _ { 1 } ) ^ { \sigma _ { 1 } } ( t _ { 2 } Z _ { 2 } ) ^ { \sigma _ { 2 } } ( t _ { 5 } Z _ { 5 } ) ^ { \sigma _ { 3 } } ( t _ { 8 } Z _ { 8 } ) ^ { \sigma _ { 4 } } $</td>
      <td>141</td>
      <td>$\tau _ { 0 } ( y ) = \sum _ { \sigma _ { 1 } = 0 } ^ { 1 } \cdot \lambda _ { e _ { 1 } = 0 } ^ { 1 } Y _ { \sigma _ { 1 } \cdots a _ { 1 } } ^ { ( 0 ) } ( t _ { 1 } Z _ { 1 } ) ^ { \sigma _ { 1 } } ( t _ { 2 } Z _ { 2 } ) ^ { \sigma _ { 2 } } ( t _ { 5 } Z _ { 5 } ) ^ { \sigma _ { 3 } } ( t _ { 8 } Z _ { 8 } ) ^ { \sigma _ { 4 } } $</td>
      <td>\tau _ { 0 } ( y ) = \sum _ { \sigma _ { 1 } = 0 } ^ { 1 } . . \sum _ { \sigma _ { 4 } = 0 } ^ { 1 } Y _ { \sigma _ { 1 } . . \sigma _ { 4 } } ^ { ( 0 ) } ( t _ { 1 } Z _ { 1 } ) ^ { \sigma _ { 1 } } ( t _ { 2 } Z _ { 2 } ) ^ { \sigma _ { 2 } } ( t _ { 5 } Z _ { 5 } ) ^ { \sigma _ { 3 } } ( t _ { 8 } Z _ { 8 } ) ^ { \sigma _ { 4 } }</td>
      <td>\tau _ { 0 } ( y ) = \sum _ { \sigma _ { 1 } = 0 } ^ { 1 } \cdot \lambda _ { e _ { 1 } = 0 } ^ { 1 } Y _ { \sigma _ { 1 } \cdots a _ { 1 } } ^ { ( 0 ) } ( t _ { 1 } Z _ { 1 } ) ^ { \sigma _ { 1 } } ( t _ { 2 } Z _ { 2 } ) ^ { \sigma _ { 2 } } ( t _ { 5 } Z _ { 5 } ) ^ { \sigma _ { 3 } } ( t _ { 8 } Z _ { 8 } ) ^ { \sigma _ { 4 } }</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.068966</td>
      <td>29</td>
      <td>$h ^ { \mu } ( P , \rho n - k ) \equiv e ^ { ( a ) \mu } s ( P , k ) \, $</td>
      <td>28</td>
      <td>$h ^ { \mu } ( P , \rho m - k ) \equiv e ^ { ( a ) \mu } s ( P , k ) $</td>
      <td>h ^ { \mu } ( P , \rho n - k ) \equiv e ^ { ( a ) \mu } s ( P , k ) \,</td>
      <td>h ^ { \mu } ( P , \rho m - k ) \equiv e ^ { ( a ) \mu } s ( P , k )</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.008403</td>
      <td>119</td>
      <td>$0 = 4 a \left( \frac { d \bar { \phi } } { d A } \right) ^ { 2 } - \frac { 2 } { P ( A ) } \frac { d P ( A ) } { d A } \frac { d \bar { \phi } } { d A } - \frac { \bar { f } _ { 0 } ^ { 2 } } { 2 P ^ { 2 } ( A ) } + \frac { 2 \bar { Q } + \mu e ^ { - 4 a \bar { \phi } } / \bar { Q } } { 4 P ( A ) } , $</td>
      <td>119</td>
      <td>$0 = 4 a \left( \frac { d \bar { \phi } } { d A } \right) ^ { 2 } - \frac { 2 } { P ( A ) } \frac { d P ( A ) } { d A } \frac { d \bar { \phi } } { d A } - \frac { \bar { f } _ { 0 } ^ { 2 } } { 2 P ^ { 2 } ( A ) } + \frac { 2 \bar { Q } + \mu e ^ { - 4 a \vec { \phi } } / \bar { Q } } { 4 P ( A ) } , $</td>
      <td>0 = 4 a \left( \frac { d \bar { \phi } } { d A } \right) ^ { 2 } - \frac { 2 } { P ( A ) } \frac { d P ( A ) } { d A } \frac { d \bar { \phi } } { d A } - \frac { \bar { f } _ { 0 } ^ { 2 } } { 2 P ^ { 2 } ( A ) } + \frac { 2 \bar { Q } + \mu e ^ { - 4 a \bar { \phi } } / \bar { Q } } { 4 P ( A ) } ,</td>
      <td>0 = 4 a \left( \frac { d \bar { \phi } } { d A } \right) ^ { 2 } - \frac { 2 } { P ( A ) } \frac { d P ( A ) } { d A } \frac { d \bar { \phi } } { d A } - \frac { \bar { f } _ { 0 } ^ { 2 } } { 2 P ^ { 2 } ( A ) } + \frac { 2 \bar { Q } + \mu e ^ { - 4 a \vec { \phi } } / \bar { Q } } { 4 P ( A ) } ,</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.081633</td>
      <td>49</td>
      <td>${ \cal M } = \left[ { \frac { S U ( 1 , 1 ) } { U ( 1 ) } } \otimes { \frac { S U ( 1 , 1 ) } { U ( 1 ) } } \right] ^ { 3 } . $</td>
      <td>45</td>
      <td>${ \cal M } = \left[ \frac { S U ( 1 , 1 ) } { U ( 1 ) } \otimes \frac { S U ( 1 , 1 ) } { U ( 1 ) } \right] ^ { 3 } . $</td>
      <td>{ \cal M } = \left[ { \frac { S U ( 1 , 1 ) } { U ( 1 ) } } \otimes { \frac { S U ( 1 , 1 ) } { U ( 1 ) } } \right] ^ { 3 } .</td>
      <td>{ \cal M } = \left[ \frac { S U ( 1 , 1 ) } { U ( 1 ) } \otimes \frac { S U ( 1 , 1 ) } { U ( 1 ) } \right] ^ { 3 } .</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.000000</td>
      <td>23</td>
      <td>$( ( \Gamma ^ { ( i n t ) } ) ^ { 2 } = 1 ) | \Phi \rangle , $</td>
      <td>23</td>
      <td>$( ( \Gamma ^ { ( i n t ) } ) ^ { 2 } = 1 ) | \Phi \rangle , $</td>
      <td>( ( \Gamma ^ { ( i n t ) } ) ^ { 2 } = 1 ) | \Phi \rangle ,</td>
      <td>( ( \Gamma ^ { ( i n t ) } ) ^ { 2 } = 1 ) | \Phi \rangle ,</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.000000</td>
      <td>27</td>
      <td>${ \cal L } = { \cal L } ^ { ( 2 ) } + { \cal L } ^ { ( 4 ) } , $</td>
      <td>27</td>
      <td>${ \cal L } = { \cal L } ^ { ( 2 ) } + { \cal L } ^ { ( 4 ) } , $</td>
      <td>{ \cal L } = { \cal L } ^ { ( 2 ) } + { \cal L } ^ { ( 4 ) } ,</td>
      <td>{ \cal L } = { \cal L } ^ { ( 2 ) } + { \cal L } ^ { ( 4 ) } ,</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.117647</td>
      <td>85</td>
      <td>$E _ { h y p } \equiv { \frac { 3 2 \pi } { 9 } } \alpha _ { s } ( \mu ) { \frac { \left| \psi ( 0 ) \right| ^ { 2 } } { m _ { h } m _ { n } } } \; \propto \; \alpha _ { s } ( \mu ) \, { \frac { \mu b } { m _ { h } m _ { n } } } $</td>
      <td>79</td>
      <td>$E _ { l h p } \equiv \frac { 3 2 \pi } { 9 } \alpha _ { s } ( \mu ) \frac { | \psi ( 0 ) | ^ { 2 } } { m _ { h } m _ { n } } \; \propto \; \alpha _ { s } ( \mu ) \, \frac { \mu b } { m _ { h } m _ { n } } $</td>
      <td>E _ { h y p } \equiv { \frac { 3 2 \pi } { 9 } } \alpha _ { s } ( \mu ) { \frac { \left| \psi ( 0 ) \right| ^ { 2 } } { m _ { h } m _ { n } } } \; \propto \; \alpha _ { s } ( \mu ) \, { \frac { \mu b } { m _ { h } m _ { n } } }</td>
      <td>E _ { l h p } \equiv \frac { 3 2 \pi } { 9 } \alpha _ { s } ( \mu ) \frac { | \psi ( 0 ) | ^ { 2 } } { m _ { h } m _ { n } } \; \propto \; \alpha _ { s } ( \mu ) \, \frac { \mu b } { m _ { h } m _ { n } }</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.000000</td>
      <td>122</td>
      <td>$\tilde { a } _ { \tilde { n } } ^ { \alpha } = \int _ { 0 } ^ { \pi } ~ \frac { d \sigma _ { - } } { \pi } \operatorname { e x p } \left[ 4 i \tilde { n } \frac { e _ { \mu } X _ { R } ^ { \mu } ( \sigma _ { - } ) } { e _ { \mu } P _ { R } ^ { \mu } } \right] \xi _ { i } ^ { \alpha } \partial _ { - } X _ { R } ^ { i } ( \sigma _ { - } ) $</td>
      <td>122</td>
      <td>$\tilde { a } _ { \tilde { n } } ^ { \alpha } = \int _ { 0 } ^ { \pi } ~ \frac { d \sigma _ { - } } { \pi } \operatorname { e x p } \left[ 4 i \tilde { n } \frac { e _ { \mu } X _ { R } ^ { \mu } ( \sigma _ { - } ) } { e _ { \mu } P _ { R } ^ { \mu } } \right] \xi _ { i } ^ { \alpha } \partial _ { - } X _ { R } ^ { i } ( \sigma _ { - } ) $</td>
      <td>\tilde { a } _ { \tilde { n } } ^ { \alpha } = \int _ { 0 } ^ { \pi } ~ \frac { d \sigma _ { - } } { \pi } \operatorname { e x p } \left[ 4 i \tilde { n } \frac { e _ { \mu } X _ { R } ^ { \mu } ( \sigma _ { - } ) } { e _ { \mu } P _ { R } ^ { \mu } } \right] \xi _ { i } ^ { \alpha } \partial _ { - } X _ { R } ^ { i } ( \sigma _ { - } )</td>
      <td>\tilde { a } _ { \tilde { n } } ^ { \alpha } = \int _ { 0 } ^ { \pi } ~ \frac { d \sigma _ { - } } { \pi } \operatorname { e x p } \left[ 4 i \tilde { n } \frac { e _ { \mu } X _ { R } ^ { \mu } ( \sigma _ { - } ) } { e _ { \mu } P _ { R } ^ { \mu } } \right] \xi _ { i } ^ { \alpha } \partial _ { - } X _ { R } ^ { i } ( \sigma _ { - } )</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.093023</td>
      <td>43</td>
      <td>$G _ { \mu } ^ { \pm } ( x , \varepsilon ) = \frac 1 2 ( G _ { \mu } ( x , \varepsilon ) \pm G _ { \mu } ( x , - \varepsilon ) ) , $</td>
      <td>47</td>
      <td>$G _ { \mu } ^ { \pm } ( x , \varepsilon ) = \frac { 1 } { 2 } ( G _ { \mu } ( x , \varepsilon ) \pm G _ { \mu } ( x , - \varepsilon ) ) , $</td>
      <td>G _ { \mu } ^ { \pm } ( x , \varepsilon ) = \frac 1 2 ( G _ { \mu } ( x , \varepsilon ) \pm G _ { \mu } ( x , - \varepsilon ) ) ,</td>
      <td>G _ { \mu } ^ { \pm } ( x , \varepsilon ) = \frac { 1 } { 2 } ( G _ { \mu } ( x , \varepsilon ) \pm G _ { \mu } ( x , - \varepsilon ) ) ,</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.000000</td>
      <td>43</td>
      <td>$\Theta _ { t } ( z ) = ( z ; t ) _ { \infty } ( t z ^ { - 1 } ; t ) _ { \infty } ( t ; t ) _ { \infty } , $</td>
      <td>43</td>
      <td>$\Theta _ { t } ( z ) = ( z ; t ) _ { \infty } ( t z ^ { - 1 } ; t ) _ { \infty } ( t ; t ) _ { \infty } , $</td>
      <td>\Theta _ { t } ( z ) = ( z ; t ) _ { \infty } ( t z ^ { - 1 } ; t ) _ { \infty } ( t ; t ) _ { \infty } ,</td>
      <td>\Theta _ { t } ( z ) = ( z ; t ) _ { \infty } ( t z ^ { - 1 } ; t ) _ { \infty } ( t ; t ) _ { \infty } ,</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.011236</td>
      <td>89</td>
      <td>$d s _ { C F T } ^ { 2 } = \operatorname* { l i m } _ { r \to \infty } \left[ { \frac { l ^ { 2 } } { r ^ { 2 } } } d s _ { n + 2 } ^ { 2 } \right] = - d t ^ { 2 } + l ^ { 2 } \gamma _ { i j } d x ^ { i } d x ^ { j } . $</td>
      <td>89</td>
      <td>$d s _ { C F T } ^ { 2 } = \operatorname* { l i m } _ { r \rightarrow \infty } \left[ { \frac { l ^ { 2 } } { r ^ { 2 } } } d s _ { n + 2 } ^ { 2 } \right] = - d t ^ { 2 } + l ^ { 2 } \gamma _ { i j } d x ^ { i } d x ^ { j } . $</td>
      <td>d s _ { C F T } ^ { 2 } = \operatorname* { l i m } _ { r \to \infty } \left[ { \frac { l ^ { 2 } } { r ^ { 2 } } } d s _ { n + 2 } ^ { 2 } \right] = - d t ^ { 2 } + l ^ { 2 } \gamma _ { i j } d x ^ { i } d x ^ { j } .</td>
      <td>d s _ { C F T } ^ { 2 } = \operatorname* { l i m } _ { r \rightarrow \infty } \left[ { \frac { l ^ { 2 } } { r ^ { 2 } } } d s _ { n + 2 } ^ { 2 } \right] = - d t ^ { 2 } + l ^ { 2 } \gamma _ { i j } d x ^ { i } d x ^ { j } .</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.089286</td>
      <td>56</td>
      <td>$\Lambda [ A ^ { I } , J ] \; = \; \varepsilon ^ { k } \frac { 1 } { \Delta } \left[ ( \partial _ { 0 } A _ { k } ^ { I } ) + \partial _ { k } J _ { 0 } \right] \; . $</td>
      <td>58</td>
      <td>$\Lambda [ A ^ { I } , J ] ~ = ~ \varepsilon ^ { k } { \frac { 1 } { \Delta } } \left[ ( \partial _ { 0 } A _ { k } ^ { I } ) + \partial _ { k } J _ { 0 } \right] ~ . $</td>
      <td>\Lambda [ A ^ { I } , J ] \; = \; \varepsilon ^ { k } \frac { 1 } { \Delta } \left[ ( \partial _ { 0 } A _ { k } ^ { I } ) + \partial _ { k } J _ { 0 } \right] \; .</td>
      <td>\Lambda [ A ^ { I } , J ] ~ = ~ \varepsilon ^ { k } { \frac { 1 } { \Delta } } \left[ ( \partial _ { 0 } A _ { k } ^ { I } ) + \partial _ { k } J _ { 0 } \right] ~ .</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.108108</td>
      <td>37</td>
      <td>$\widetilde \psi ( x ^ { \alpha } , z ^ { \prime } ) = \widetilde \Psi ( x ^ { \alpha } , L _ { 6 } + z ^ { \prime } ) $</td>
      <td>41</td>
      <td>$\widetilde { \psi } ( x ^ { \alpha } , z ^ { \prime } ) = \widetilde { \Psi } ( x ^ { \alpha } , L _ { 6 } + z ^ { \prime } ) $</td>
      <td>\widetilde \psi ( x ^ { \alpha } , z ^ { \prime } ) = \widetilde \Psi ( x ^ { \alpha } , L _ { 6 } + z ^ { \prime } )</td>
      <td>\widetilde { \psi } ( x ^ { \alpha } , z ^ { \prime } ) = \widetilde { \Psi } ( x ^ { \alpha } , L _ { 6 } + z ^ { \prime } )</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.157895</td>
      <td>38</td>
      <td>$M _ { \sigma } = 4 6 3 \; \mathrm { M e V } \; \; , \; \; \Gamma _ { \sigma } = 3 9 3 \; \mathrm { M e V } . $</td>
      <td>38</td>
      <td>$M _ { \sigma } = 4 6 3 \ \mathrm { M e V } \ \ , \ \ \Gamma _ { \sigma } = 3 9 3 \ \mathrm { M e V } . $</td>
      <td>M _ { \sigma } = 4 6 3 \; \mathrm { M e V } \; \; , \; \; \Gamma _ { \sigma } = 3 9 3 \; \mathrm { M e V } .</td>
      <td>M _ { \sigma } = 4 6 3 \ \mathrm { M e V } \ \ , \ \ \Gamma _ { \sigma } = 3 9 3 \ \mathrm { M e V } .</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.000000</td>
      <td>39</td>
      <td>$W ^ { ( m ) } = e ^ { Q } \, ( X ^ { + } W ^ { - } + X ^ { - } W ^ { + } ) \; . $</td>
      <td>39</td>
      <td>$W ^ { ( m ) } = e ^ { Q } \, ( X ^ { + } W ^ { - } + X ^ { - } W ^ { + } ) \; . $</td>
      <td>W ^ { ( m ) } = e ^ { Q } \, ( X ^ { + } W ^ { - } + X ^ { - } W ^ { + } ) \; .</td>
      <td>W ^ { ( m ) } = e ^ { Q } \, ( X ^ { + } W ^ { - } + X ^ { - } W ^ { + } ) \; .</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.000000</td>
      <td>25</td>
      <td>$( F , F ) = \int _ { M ^ { D } } T r ( \tilde { F } \wedge F ) $</td>
      <td>25</td>
      <td>$( F , F ) = \int _ { M ^ { D } } T r ( \tilde { F } \wedge F ) $</td>
      <td>( F , F ) = \int _ { M ^ { D } } T r ( \tilde { F } \wedge F )</td>
      <td>( F , F ) = \int _ { M ^ { D } } T r ( \tilde { F } \wedge F )</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.028571</td>
      <td>35</td>
      <td>$Z = \int { \cal D } ( g , \Phi ) \; \operatorname { e x p } [ - I _ { E } ( g , \Phi ) / \hbar ] . $</td>
      <td>35</td>
      <td>$Z = \int { \cal D } ( g , \Phi ) \ \operatorname { e x p } [ - I _ { E } ( g , \Phi ) / \hbar ] . $</td>
      <td>Z = \int { \cal D } ( g , \Phi ) \; \operatorname { e x p } [ - I _ { E } ( g , \Phi ) / \hbar ] .</td>
      <td>Z = \int { \cal D } ( g , \Phi ) \ \operatorname { e x p } [ - I _ { E } ( g , \Phi ) / \hbar ] .</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.000000</td>
      <td>46</td>
      <td>$\bar { \alpha } _ { R G } ^ { ( 1 ) } ( x , 0 , \alpha ) = \frac { \alpha } { 1 - \frac { \alpha } { 3 \pi } \cdot \operatorname { l n } x } $</td>
      <td>46</td>
      <td>$\bar { \alpha } _ { R G } ^ { ( 1 ) } ( x , 0 , \alpha ) = \frac { \alpha } { 1 - \frac { \alpha } { 3 \pi } \cdot \operatorname { l n } x } $</td>
      <td>\bar { \alpha } _ { R G } ^ { ( 1 ) } ( x , 0 , \alpha ) = \frac { \alpha } { 1 - \frac { \alpha } { 3 \pi } \cdot \operatorname { l n } x }</td>
      <td>\bar { \alpha } _ { R G } ^ { ( 1 ) } ( x , 0 , \alpha ) = \frac { \alpha } { 1 - \frac { \alpha } { 3 \pi } \cdot \operatorname { l n } x }</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.082353</td>
      <td>85</td>
      <td>$\Psi _ { ( \omega _ { 0 } , \vec { k } _ { 0 } ) } = e ^ { - i \vec { k } _ { 0 } { \cdot } \vec { x } } \left[ \int e ^ { - i \Delta \vec { k } { \cdot } \vec { x } } e ^ { i \Delta \omega t } d \mu \right] e ^ { i \omega _ { 0 } t } \! $</td>
      <td>80</td>
      <td>$\Psi _ { ( \omega _ { 0 } , \vec { k } _ { 0 } ) } = e ^ { - i \vec { k } _ { 0 } \cdot \vec { x } } \left[ \int e ^ { - i \Delta \vec { k } \cdot \vec { e } } e ^ { i \omega \omega t } d \mu \right] e ^ { i \omega _ { 0 } t } $</td>
      <td>\Psi _ { ( \omega _ { 0 } , \vec { k } _ { 0 } ) } = e ^ { - i \vec { k } _ { 0 } { \cdot } \vec { x } } \left[ \int e ^ { - i \Delta \vec { k } { \cdot } \vec { x } } e ^ { i \Delta \omega t } d \mu \right] e ^ { i \omega _ { 0 } t } \!</td>
      <td>\Psi _ { ( \omega _ { 0 } , \vec { k } _ { 0 } ) } = e ^ { - i \vec { k } _ { 0 } \cdot \vec { x } } \left[ \int e ^ { - i \Delta \vec { k } \cdot \vec { e } } e ^ { i \omega \omega t } d \mu \right] e ^ { i \omega _ { 0 } t }</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.000000</td>
      <td>39</td>
      <td>$\left( \begin{array} { c c } { a } &amp; { b } \\ { c } &amp; { d } \\ \end{array} \right) \left( \begin{array} { c } { \tau } \\ { 1 } \\ \end{array} \right) $</td>
      <td>39</td>
      <td>$\left( \begin{array} { c c } { a } &amp; { b } \\ { c } &amp; { d } \\ \end{array} \right) \left( \begin{array} { c } { \tau } \\ { 1 } \\ \end{array} \right) $</td>
      <td>\left( \begIn{array} { c c } { a } &amp; { b } \\ { c } &amp; { d } \\ \end{array} \right) \left( \begIn{array} { c } { \tau } \\ { 1 } \\ \end{array} \right)</td>
      <td>\left( \begIn{array} { c c } { a } &amp; { b } \\ { c } &amp; { d } \\ \end{array} \right) \left( \begIn{array} { c } { \tau } \\ { 1 } \\ \end{array} \right)</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.177215</td>
      <td>79</td>
      <td>$S _ { \varepsilon , \lambda } = { \mathrm T } \operatorname { e x p } \left( - { \mathrm i } \lambda \, \int _ { - \infty } ^ { \infty } { \mathrm d } t \int { \mathrm d } ^ { 3 } x \, \operatorname { e x p } ( - \varepsilon | t | ) \, { \mathcal H } _ { \mathrm I } ( x ) \right) $</td>
      <td>80</td>
      <td>$S _ { \varepsilon , \lambda } = \mathrm { T } \operatorname { e x p } \left( - \mathrm { i } \lambda \int _ { - \infty } ^ { \infty } \mathrm { d } t \int \mathrm { d } ^ { 3 } x ~ \operatorname { e x p } ( - \varepsilon | t | ) \, { \cal H } _ { \mathrm { l } } ( x ) \right) $</td>
      <td>S _ { \varepsilon , \lambda } = { \mathrm T } \operatorname { e x p } \left( - { \mathrm i } \lambda \, \int _ { - \infty } ^ { \infty } { \mathrm d } t \int { \mathrm d } ^ { 3 } x \, \operatorname { e x p } ( - \varepsilon | t | ) \, { \mathcal H } _ { \mathrm I } ( x ) \right)</td>
      <td>S _ { \varepsilon , \lambda } = \mathrm { T } \operatorname { e x p } \left( - \mathrm { i } \lambda \int _ { - \infty } ^ { \infty } \mathrm { d } t \int \mathrm { d } ^ { 3 } x ~ \operatorname { e x p } ( - \varepsilon | t | ) \, { \cal H } _ { \mathrm { l } } ( x ) \right)</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.112500</td>
      <td>80</td>
      <td>$\xi _ { r } ( t ) \ = \ e ^ { - i r ( t - t _ { i } ) / \rho } \xi _ { i r } \ , \ \xi _ { r } ^ { * } \ = \ e ^ { - i r ( t _ { f } - t ) / \rho } \xi _ { f r } ^ { * } \ , $</td>
      <td>80</td>
      <td>$\xi _ { r } ( t ) \; = \; e ^ { - i r ( t - t _ { i } ) / \rho } \xi _ { i r } \; , \; \xi _ { r } ^ { \ast } \; = \; e ^ { - i r ( t _ { f } - t ) / \rho } \xi _ { f r } ^ { \ast } \; , $</td>
      <td>\xi _ { r } ( t ) \ = \ e ^ { - i r ( t - t _ { i } ) / \rho } \xi _ { i r } \ , \ \xi _ { r } ^ { * } \ = \ e ^ { - i r ( t _ { f } - t ) / \rho } \xi _ { f r } ^ { * } \ ,</td>
      <td>\xi _ { r } ( t ) \; = \; e ^ { - i r ( t - t _ { i } ) / \rho } \xi _ { i r } \; , \; \xi _ { r } ^ { \ast } \; = \; e ^ { - i r ( t _ { f } - t ) / \rho } \xi _ { f r } ^ { \ast } \; ,</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.009259</td>
      <td>108</td>
      <td>$\vec { T } _ { 0 } ^ { 1 } \vec { T } _ { 1 } ^ { 1 } \; = \; s _ { - 1 } ^ { r - 1 } s _ { 1 } ^ { r - 1 } f _ { - 1 } ^ { 1 } f _ { 1 } ^ { 1 } \, \vec { I } + s _ { 0 } ^ { r - 1 } f _ { 0 } ^ { 1 } \, \vec { T } _ { 0 } ^ { 2 } $</td>
      <td>108</td>
      <td>$\vec { T } _ { 0 } ^ { 1 } \vec { T } _ { 1 } ^ { 1 } \; = \; s _ { - 1 } ^ { r - 1 } s _ { 1 } ^ { r - 1 } f _ { - 1 } ^ { 1 } f _ { 1 } ^ { 1 } \; \vec { I } + s _ { 0 } ^ { r - 1 } f _ { 0 } ^ { 1 } \, \vec { T } _ { 0 } ^ { 2 } $</td>
      <td>\vec { T } _ { 0 } ^ { 1 } \vec { T } _ { 1 } ^ { 1 } \; = \; s _ { - 1 } ^ { r - 1 } s _ { 1 } ^ { r - 1 } f _ { - 1 } ^ { 1 } f _ { 1 } ^ { 1 } \, \vec { I } + s _ { 0 } ^ { r - 1 } f _ { 0 } ^ { 1 } \, \vec { T } _ { 0 } ^ { 2 }</td>
      <td>\vec { T } _ { 0 } ^ { 1 } \vec { T } _ { 1 } ^ { 1 } \; = \; s _ { - 1 } ^ { r - 1 } s _ { 1 } ^ { r - 1 } f _ { - 1 } ^ { 1 } f _ { 1 } ^ { 1 } \; \vec { I } + s _ { 0 } ^ { r - 1 } f _ { 0 } ^ { 1 } \, \vec { T } _ { 0 } ^ { 2 }</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.058824</td>
      <td>68</td>
      <td>$2 \, g _ { u v } \, \dot { u } \, \dot { v } = - ( { \frac { p _ { x } ^ { 2 } } { g _ { x x } } } + { \frac { p _ { y } ^ { 2 } } { g _ { y y } } } ) . $</td>
      <td>64</td>
      <td>$2 \, g _ { u v } \, \dot { u } \, \dot { v } = - ( \frac { p _ { x } ^ { 2 } } { g _ { x x } } + \frac { p _ { y } ^ { 2 } } { g _ { y y } } ) . $</td>
      <td>2 \, g _ { u v } \, \dot { u } \, \dot { v } = - ( { \frac { p _ { x } ^ { 2 } } { g _ { x x } } } + { \frac { p _ { y } ^ { 2 } } { g _ { y y } } } ) .</td>
      <td>2 \, g _ { u v } \, \dot { u } \, \dot { v } = - ( \frac { p _ { x } ^ { 2 } } { g _ { x x } } + \frac { p _ { y } ^ { 2 } } { g _ { y y } } ) .</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.123711</td>
      <td>97</td>
      <td>$\theta _ { 1 3 } = \Gamma _ { 1 } e ^ { - \sum \lambda ^ { n } t _ { n } } \; \; ; \; \; \theta _ { 2 3 } = \Gamma _ { 2 } e ^ { - \sum \lambda ^ { n } t _ { n } } \; \; ; \; \; \theta _ { 3 3 } = \psi _ { B A } e ^ { - \sum \lambda ^ { n } t _ { n } } \, , $</td>
      <td>100</td>
      <td>$\theta _ { 1 3 } = \Gamma _ { 1 } e ^ { - \sum \lambda ^ { n } t _ { n } } \ \ ; \ \ \theta _ { 2 3 } = \Gamma _ { 2 } e ^ { - \sum \lambda ^ { n } t _ { n } } \ \ ; \ \ \theta _ { 3 3 } = \psi _ { B A } e ^ { - \sum _ { \lambda } ^ { n } l _ { n } } \, , $</td>
      <td>\theta _ { 1 3 } = \Gamma _ { 1 } e ^ { - \sum \lambda ^ { n } t _ { n } } \; \; ; \; \; \theta _ { 2 3 } = \Gamma _ { 2 } e ^ { - \sum \lambda ^ { n } t _ { n } } \; \; ; \; \; \theta _ { 3 3 } = \psi _ { B A } e ^ { - \sum \lambda ^ { n } t _ { n } } \, ,</td>
      <td>\theta _ { 1 3 } = \Gamma _ { 1 } e ^ { - \sum \lambda ^ { n } t _ { n } } \ \ ; \ \ \theta _ { 2 3 } = \Gamma _ { 2 } e ^ { - \sum \lambda ^ { n } t _ { n } } \ \ ; \ \ \theta _ { 3 3 } = \psi _ { B A } e ^ { - \sum _ { \lambda } ^ { n } l _ { n } } \, ,</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.067416</td>
      <td>89</td>
      <td>$f ^ { \prime } ( s ) - f ^ { \prime } ( u ) + \frac { 1 } { 3 } ( 2 g ^ { \prime } ( s ) + g ^ { \prime } ( u ) ) + \frac { 1 } { 3 } ( h ( u ) + 2 h ( t ) ) - \frac { 1 } { 3 } ( s - t ) h ^ { \prime } ( u ) = 0 , $</td>
      <td>95</td>
      <td>$f ^ { \prime } ( s ) - f ^ { \prime } ( u ) + { \frac { 1 } { 3 } } ( 2 g ^ { \prime } ( s ) + g ^ { \prime } ( u ) ) + { \frac { 1 } { 3 } } ( h ( u ) + 2 h ( t ) ) - { \frac { 1 } { 3 } } ( s - t ) h ^ { \prime } ( u ) = 0 , $</td>
      <td>f ^ { \prime } ( s ) - f ^ { \prime } ( u ) + \frac { 1 } { 3 } ( 2 g ^ { \prime } ( s ) + g ^ { \prime } ( u ) ) + \frac { 1 } { 3 } ( h ( u ) + 2 h ( t ) ) - \frac { 1 } { 3 } ( s - t ) h ^ { \prime } ( u ) = 0 ,</td>
      <td>f ^ { \prime } ( s ) - f ^ { \prime } ( u ) + { \frac { 1 } { 3 } } ( 2 g ^ { \prime } ( s ) + g ^ { \prime } ( u ) ) + { \frac { 1 } { 3 } } ( h ( u ) + 2 h ( t ) ) - { \frac { 1 } { 3 } } ( s - t ) h ^ { \prime } ( u ) = 0 ,</td>
    </tr>
  </tbody>
</table>
</div>


