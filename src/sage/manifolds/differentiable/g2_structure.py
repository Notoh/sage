from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING, overload

from sage.manifolds.differentiable.diff_form import DiffForm, DiffFormParal
from sage.manifolds.differentiable.metric import PseudoRiemannianMetric, PseudoRiemannianMetricParal

if TYPE_CHECKING:
    from sage.manifolds.differentiable.diff_map import DiffMap
    from sage.manifolds.differentiable.manifold import DifferentiableManifold
    from sage.symbolic.expression import Expression
    from sage.manifolds.differentiable.vectorfield_module import VectorFieldModule
    from sage.manifolds.differentiable.vectorfield import VectorField
    from sage.manifolds.differentiable.scalarfield import DiffScalarField
    from sage.manifolds.differentiable.tensorfield_paral import TensorFieldParal
    from sage.manifolds.differentiable.tensorfield import TensorField
    from sage.manifolds.differentiable.vectorframe import VectorFrame

class G2Structure(DiffForm):
    _name: str
    _latex_name: str
    _dual = Optional[DiffForm]
    _vol_form: Optional[DiffForm]
    _metric = Optional[PseudoRiemannianMetric]

    def __init__(
        self,
        manifold: Union[DifferentiableManifold, VectorFieldModule],
        name: Optional[str] = None,
        latex_name: Optional[str] = None,
    ):
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold
        
        if name is None:
            name = "phi"
            if latex_name is None:
                latex_name = r"\varphi"
        if latex_name is None:
            latex_name = name

        DiffForm.__init__(
            self, vector_field_module, 3, name=name, latex_name=latex_name
        )
        dim = self._ambient_domain.dimension()
    
        if dim != 7:
            raise ValueError(f"G2 structure can only be defined on a 7-dimensional manifold, got dimension {dim}.")
        if not self._domain.orientation():
            raise ValueError('{} must admit an orientation'.format(self._domain))
        G2Structure._init_derived(self)
    
    def _repr_(self):
        return self._final_repr(f"G2 structure {self._name} ")

    def _new_instance(self):
        return type(self)(
            self._vmodule,
            "unnamed G2 structure",
            latex_name=r"\text{unnamed G2 structure}",
        )

    def _init_derived(self):
        DiffForm._init_derived(self)
        
        self._metric = None
        self._vol_form = None
        self._dual = None

    def _del_derived(self):
        # Delete the derived quantities from the mother class
        DiffForm._del_derived(self)

        if self._metric is not None:
            self._metric._restrictions.clear()
            self._metric._del_derived()
            self._metric = None

        if self._vol_form is not None:
            self._vol_form._restrictions.clear()
            self._vol_form._del_derived()
            self._vol_form = None

        if self._dual is not None:
            self._dual._restrictions.clear()
            self._dual._del_derived()
            self._dual = None
    
    def restrict(
        self, subdomain: DifferentiableManifold, dest_map: Optional[DiffMap] = None
    ) -> DiffForm:
        if subdomain == self._domain:
            return self

        if subdomain not in self._restrictions:
            # Construct the restriction at the tensor field level
            restriction = DiffForm.restrict(self, subdomain, dest_map=dest_map)

            # Restrictions of derived quantities
            if self._metric is not None:
                restriction._metric = self._metric.restrict(subdomain)
                if self._vol_form is not None:
                    restriction._vol_form = restriction._metric.vol_forms[0]
            if self._vol_form is not None:
                restriction._vol_form = self._vol_form.restrict(subdomain)
            if self._dual is not None:
                restriction._dual = self._dual.restrict(subdomain)

            # The restriction is ready
            self._restrictions[subdomain] = restriction
            return restriction
        return self._restrictions[subdomain]

    @staticmethod
    def wrap(
        form: DiffForm, name: Optional[str] = None, latex_name: Optional[str] = None
    ) -> G2Structure:
        if form.degree() != 3:
            raise ValueError("Only 3-forms can be G2 structures.")

        if name is None:
            name = form._name
        if latex_name is None:
            latex_name = form._latex_name

        g2_structure = form.base_module().g2_structure(name=name, latex_name=latex_name)

        for dom, rst in form._restrictions.items():
            g2_structure._restrictions[dom] = G2Structure.wrap(
                rst, name, latex_name
            )

        if isinstance(form, DiffFormParal):
            for frame in form._components:
                g2_structure._components[frame] = form._components[frame].copy()

        return g2_structure

    def dual(self) -> DiffForm:
        if self._dual is None:
            self._dual = self.hodge_dual(self.metric())
        return self._dual
    
    def metric(self) -> PseudoRiemannianMetric:
        if self._metric is None:
            self._metric = PseudoRiemannianMetric(self._vmodule, name=f"g_{self._name}", latex_name=f"g_{self._latex_name}")
        
        # TODO only do this when a restriction is added
        for domain, rst in self._restrictions.items():
            self._metric._restrictions[domain] = rst.metric()

        return self._metric
    
    def volume_form(self, contra: int = 0) -> DiffForm:
        if contra != 0:
            raise ValueError("contra must be 0 for Hodge duals taken with respect to G2 structures.")

        if self._vol_form is None:
            g = self.metric()
            if g._vol_forms[0] is None:
                self._vol_form = 1/7 * self.wedge(self.dual())
                g._vol_forms[0] = self._vol_form
            else:
                self._vol_form = g._vol_forms[0]
        return self._vol_form

    def hodge_star(self, pform: DiffForm) -> DiffForm:
        return pform.hodge_dual(self)

    # Computes P(form) = 2 * (self ^ form). 
    # This is unbearably slow, we need to use hodge_P_onf and figure that out globally
    def hodge_P(self, form: DiffForm) -> DiffForm:
        if form.degree() != 2:
            raise ValueError("Only 2-forms can be input into hodge_P")
        return 2 * (self.wedge(form)).hodge_dual(self.metric())

    # TODO SEE COMMENT IN VectorFrame
    def hodge_P_onf(self, form, frame: VectorFrame) -> DiffForm:
        if form.degree() != 2:
            raise ValueError("Only 2-forms can be input into hodge_P_onf")
        common_domain = frame.domain().intersection(self._domain)

        res = common_domain.diff_form(2)
        psi = self.dual()

        for a in range(1, 8):
            for b in range(a+1, 8):
                res[frame, a,b] = 0
                for i in range(1, 8):
                    for j in range(i+1, 8):
                        res[frame, a,b] += 2 * form[i,j] * psi[i,j,a,b]
        return res

    def form_pi_7(self, form: DiffForm, pform: Optional[DiffForm] = None) -> DiffForm:
        if form.degree() != 2 and form.degree() != 5:
            raise ValueError("Only 2-forms and 5-forms can be input into form_pi_7")
        dualize = False
        if form.degree() == 5:
            form = form.hodge_dual(self.metric())
            dualize = True
        if pform is None:
            pform = self.hodge_P(form)
        from sage.symbolic.ring import SR

        return SR(1)/6 * (2 * form - pform) if not dualize else (SR(1)/6 * (2 * form - pform)).hodge_dual(self.metric())

    def form_pi_14(self, form: DiffForm, pform: Optional[DiffForm] = None) -> DiffForm:
        if form.degree() != 2 and form.degree() != 5:
            raise ValueError("Only 2-forms and 5-forms can be input into form_pi_14")
        dualize = False
        if form.degree() == 5:
            form = form.hodge_dual(self.metric())
            dualize = True
        if pform is None:
            pform = self.hodge_P(form)
        from sage.symbolic.ring import SR

        return SR(1)/6 * (4 * form + pform) if not dualize else (SR(1)/6 * (4 * form + pform)).hodge_dual(self.metric())

class G2StructureParal(DiffFormParal, G2Structure):
    _dual: DiffFormParal
    _vol_form: DiffFormParal
    _metric: PseudoRiemannianMetricParal

    def __init__(
        self,
        manifold: Union[VectorFieldModule, DifferentiableManifold],
        name: Optional[str],
        latex_name: Optional[str] = None,
    ):
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold

        if name is None:
            name = "phi"

        DiffFormParal.__init__(
            self, vector_field_module, 3, name=name, latex_name=latex_name
        )

        # Check that manifold is even dimensional
        dim = self._ambient_domain.dimension()
    
        if dim != 7:
            raise ValueError(f"G2 structure can only be defined on a 7-dimensional manifold, got dimension {dim}.")

        # Initialization of derived quantities
        G2StructureParal._init_derived(self)

    def _init_derived(self):
        # Initialization of quantities pertaining to mother classes
        DiffFormParal._init_derived(self)
        G2Structure._init_derived(self)

    def _del_derived(self, del_restrictions: bool = True):
        DiffFormParal._del_derived(self, del_restrictions=del_restrictions)

        if self._dual is not None:
            self._dual._components.clear()
            self._dual._del_derived()
        if self._metric is not None:
            self._metric._components.clear()
            self._metric._del_derived()
        if self._vol_form is not None:
            self._vol_form._components.clear()
            self._vol_form._del_derived()

        G2Structure._del_derived(self)

    def restrict(
        self, subdomain: DifferentiableManifold, dest_map: Optional[DiffMap] = None
    ) -> G2StructureParal:
        if subdomain == self._domain:
            return self
        if subdomain not in self._restrictions:
            # Construct the restriction at the tensor field level:
            resu = DiffFormParal.restrict(self, subdomain, dest_map=dest_map)

            self._restrictions[subdomain] = G2StructureParal.wrap(resu)
        return self._restrictions[subdomain]

    def metric(self) -> PseudoRiemannianMetricParal:
        if self._metric is None:
            self._metric = PseudoRiemannianMetricParal(self._vmodule, name=f"g_{self._name}", latex_name=f"g_{self._latex_name}")

        from sage.matrix.constructor import matrix
        from sage.symbolic.ring import SR

        fmodule = self._fmodule
        si = fmodule._sindex
        nsi = si + fmodule.rank()

        for frame in self._components:
            if frame not in self._metric._components:

                ip_phi = {i : frame[i].interior_product(self) for i in range(si, nsi)}

                def hd(omega, f) -> DiffFormParal:
                    res = omega
                    for i in range(si, nsi):
                        res = f[i].interior_product(res)
                    return res

                comp_B = {}

                for i in range(si, nsi):
                    for j in range(i, nsi):
                        comp_B[(i, j)] = -SR(1)/6  * hd(ip_phi[i].wedge(ip_phi[j]).wedge(self), frame).expr()
                        if i != j:
                            comp_B[(j, i)] = comp_B[(i, j)]
                B = matrix([[comp_B[(i, j)] for j in range(si, nsi)] for i in range(si, nsi)])
                det = B.determinant()

                if det == 0:
                    raise ValueError("The G2 structure is degenerate, the associated metric cannot be defined.")
                scale = det ** (-SR(1)/9)
                for i in range(si, nsi):
                    for j in range(i, nsi):
                        self._metric[frame, i, j] = scale * comp_B[(i, j)]

        return self._metric

