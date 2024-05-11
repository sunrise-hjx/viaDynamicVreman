/*---------------------------------------------------------------------------*\
viaDynamicVreman - Implementation of the dynamic Vreman
                     SGS model.

Copyright Information
    Copyright (C) 1991-2009 OpenCFD Ltd.
    Copyright (C) 2010-2021 Alberto Passalacqua 

License
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/

#include "viaDynamicVreman.H"
#include "fvOptions.H"
#include "syncTools.H"
#include "volFieldsFwd.H"
#include "fvcAverage.H"
#include "fvc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void viaDynamicVreman<BasicTurbulenceModel>::correctNut
(
    const volTensorField& gradU
)
{
    const fvMesh& mesh = this->mesh_;
    const volScalarField& nu = this->nu();
    volScalarField deltaSqr = Foam::sqr(this->delta());
    volTensorField beta = T(gradU) & gradU;
    volScalarField Bbeta(Foam::sqr(deltaSqr));
    volScalarField oneF(deltaSqr);
    forAll(deltaSqr, cellI)
    {
        oneF[cellI] = 1.0;
        tensor bec(beta[cellI]);
        Bbeta[cellI] *= (bec.xx() * bec.yy() - bec.xy() * bec.xy() +
                         bec.xx() * bec.zz() - bec.xz() * bec.xz() +
                         bec.yy() * bec.zz() - bec.yz() * bec.yz());
    }
    volScalarField aa = gradU && gradU;
    volScalarField prodg(aa);

    forAll(aa, cellI)
    {
        if(mag(Bbeta[cellI])<SMALL) Bbeta[cellI] = 0;
        prodg[cellI] = ((mag(aa[cellI]) < SMALL) ? 0.0 : (Foam::sqrt(mag(Bbeta[cellI] / aa[cellI]))));
    }
    prodg.correctBoundaryConditions();
    volTensorField S(0.5 * (gradU + T(gradU)));
    volScalarField F_aa = filter_(aa);
    volTensorField F_a(gradU);
    volTensorField F_S = filter_(S);
    volScalarField F_prodgSS = filter_((prodg * (S && S)));
    volScalarField F_delta = filter_(this->delta());
    deltaSqr = Foam::sqr(F_delta);
    Bbeta = Foam::sqr(deltaSqr);
    forAll(deltaSqr, cellI)
    {
        tensor bec(beta[cellI]);
        Bbeta[cellI] *= (bec.xx() * bec.yy() - bec.xy() * bec.xy() +
                         bec.xx() * bec.zz() - bec.xz() * bec.xz() +
                         bec.yy() * bec.zz() - bec.yz() * bec.yz());
    }
    volScalarField prodt(aa);
    forAll(aa, cellI)
    {
        if(mag(Bbeta[cellI])<SMALL) Bbeta[cellI] = 0;
        prodt[cellI] = ((mag(aa[cellI]) < SMALL) ? 0.0 : (Foam::sqrt(mag(Bbeta[cellI] / aa[cellI]))));
    }
    prodt.correctBoundaryConditions();
    volScalarField up(F_aa - (F_a && F_a));
    volScalarField down((F_prodgSS - prodt * (F_S && F_S)));
    dimensionedScalar upS = fvc::domainIntegrate(up) / fvc::domainIntegrate(oneF);
    dimensionedScalar downS = fvc::domainIntegrate(down) / fvc::domainIntegrate(oneF);
    C_.value() = -0.5 * nu[0] * (upS.value() / (downS.value()+ SMALL));
    this->nut_ = C_ * prodg;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();
}

template<class BasicTurbulenceModel>
void viaDynamicVreman<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
viaDynamicVreman<BasicTurbulenceModel>::viaDynamicVreman
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    C_("C", (dimArea*dimTime), 0.0),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool viaDynamicVreman<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());

        return true;
    }

    return false;
}

template<class BasicTurbulenceModel>
void viaDynamicVreman<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    LESeddyViscosity<BasicTurbulenceModel>::correct();
    
    correctNut(fvc::grad(this->U_));
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //