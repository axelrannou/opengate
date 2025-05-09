/*------------------------------------
   Write something professional here
-------------------------------------*/

#include "GateRandomMultiGauss.h"
#include "G4RandomTools.hh"
#include <math.h>
#include <vector>

using namespace std;

GateRandomMultiGauss::GateRandomMultiGauss(const vector<double> &muVin,
                                           const vector<double> &sigmaMin) {
  muV = muVin;
  sigmaM = sigmaMin;
  eigVal1 = 0;
  eigVal2 = 0;

  a = sigmaM[0];
  b = sigmaM[1];
  c = sigmaM[2];
  d = sigmaM[3];
}

GateRandomMultiGauss::~GateRandomMultiGauss() {}

void GateRandomMultiGauss::eigenVal() {
  const double t1 = (a + d) / 2;
  const double t2 = sqrt(pow((a + d), 2) / 4 - a * d + b * c);
  eigVal1 = t1 + t2;
  eigVal2 = t1 - t2;
}

vector<double>
GateRandomMultiGauss::eigenVector(const double eigenValue) const {
  const double x = 1;
  const double lam = eigenValue;

  double y = x * ((-(a + c) + lam) / (b + d - lam));
  vector<double> eV = {x, y};
  eV[0] /= sqrt(pow(x, 2) + pow(y, 2));
  eV[1] /= sqrt(pow(x, 2) + pow(y, 2));

  return eV;
}

void GateRandomMultiGauss::eigenVectors() {
  // NO SAFETY CHECKS PERFORMED FOR NOW  as only symmetric matrices are used //

  /*double lam = eigenValueV; // which value?
  if (((b+d -lam) < FLT_EPSILON) && ( (a+c)-lam < FLT_EPSILON)){
      eigVec1 = {1,0};
      eigVec2 = {0,1};
      }
  else{*/
  eigVec1 = eigenVector(eigVal1);
  eigVec2 = eigenVector(eigVal2);
  //}
}

vector<double> GateRandomMultiGauss::SigmaIndex(const double x1_o,
                                                const double x2_o) const {
  const double l1 = eigVal1;
  const double l2 = eigVal2;

  const double y1 = sqrt(l1) * eigVec1[0] * x1_o + sqrt(l2) * eigVec2[0] * x2_o;
  const double y2 = sqrt(l1) * eigVec1[1] * x1_o + sqrt(l2) * eigVec2[1] * x2_o;

  vector<double> r = {y1, y2};
  return r;
}

vector<double> GateRandomMultiGauss::Fire() {
  // Generate two random numbers
  const double v1 = G4RandGauss::shoot(0., 1.);
  const double v2 = G4RandGauss::shoot(0., 1.);
  eigenVal();
  eigenVectors();

  vector<double> res = SigmaIndex(v1, v2);
  res[0] += muV[0];
  res[1] += muV[1];
  return res;
}
