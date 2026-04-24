! Minimal backend for the chromatographic peak model.
! C-callable routines:
!   peak_pdf            : PDF, no derivatives.
!   peak_pdf_jac        : PDF and derivatives with respect to physical parameters.
!   peak_pdf_jac_fitvars: PDF and derivatives with respect to transformed fit variables.
!   required_L_fitvars  : truncation order for transformed fit variables.
module chromatopeak_core
  use iso_c_binding, only: c_int, c_double
  implicit none
  private
  public :: peak_pdf, peak_pdf_jac, peak_pdf_jac_fitvars, required_L_fitvars

  integer, parameter :: dp = c_double
  real(dp), parameter :: SQRT2   = 1.414213562373095048801688724209698079_dp
  real(dp), parameter :: SQRTPI  = 1.772453850905516027298167483341145183_dp
  real(dp), parameter :: SQRT2PI = 2.506628274631000502415765284811045253_dp
  real(dp), parameter :: ERFCX_THR = 26.0_dp
  real(dp), parameter :: SP_THR    = 30.0_dp
contains

  pure function softplus(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    if (x > SP_THR) then
      y = x
    else if (x < -SP_THR) then
      y = exp(x)
    else
      y = log(1.0_dp + exp(x))
    end if
  end function softplus

  pure function sigmoid(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    if (x > SP_THR) then
      y = 1.0_dp
    else if (x < -SP_THR) then
      y = exp(x)
    else
      y = 1.0_dp / (1.0_dp + exp(-x))
    end if
  end function sigmoid

  subroutine xi_to_physical(xi, nxi, M, params)
    integer, intent(in) :: nxi, M
    real(dp), intent(in)  :: xi(nxi)
    real(dp), intent(out) :: params(nxi)
    integer  :: j
    real(dp) :: cumtheta

    params(1) = xi(1)
    params(2) = softplus(xi(2))
    do j = 1, M
      params(2 + j) = softplus(xi(2 + j))
    end do
    cumtheta = 0.0_dp
    do j = 1, M
      cumtheta = cumtheta + softplus(xi(2 + M + j))
      params(2 + M + j) = cumtheta
    end do
  end subroutine xi_to_physical

  subroutine peak_pdf(t, nt, params, npar, M, L, f_out) bind(c, name='peak_pdf')
    integer(c_int), intent(in), value :: nt, npar, M, L
    real(dp), intent(in)  :: t(nt), params(npar)
    real(dp), intent(out) :: f_out(nt)

    real(dp) :: mu_G, sigma_G, Lambda_total, expL
    real(dp) :: Lambda(M), theta(M), aj(M)
    real(dp) :: bcoeff(L), Omega(0:L)
    integer  :: j

    mu_G    = params(1)
    sigma_G = params(2)
    Lambda(1:M) = params(3:2+M)
    theta(1:M)  = params(3+M:2+2*M)
    Lambda_total = sum(Lambda(1:M))
    expL = exp(-Lambda_total)

    do j = 1, M
      aj(j) = 1.0_dp - theta(1) / theta(j)
    end do

    call compute_bm(aj, Lambda, M, L, bcoeff)
    call compute_Omega(bcoeff, L, Omega)
    call accumulate_pdf_vectorized(t, nt, mu_G, sigma_G, theta(1), L, Omega, expL, f_out)
  end subroutine peak_pdf

  subroutine peak_pdf_jac(t, nt, params, npar, M, L, f_out, jac_out) bind(c, name='peak_pdf_jac')
    integer(c_int), intent(in), value :: nt, npar, M, L
    real(dp), intent(in)  :: t(nt), params(npar)
    real(dp), intent(out) :: f_out(nt), jac_out(npar, nt)

    real(dp) :: mu_G, sigma_G, Lambda_total, expL
    real(dp) :: Lambda(M), theta(M), aj(M)
    real(dp) :: bcoeff(L), Omega(0:L), db(L, 2*M), dOmega(0:L, 2*M)
    integer  :: j

    mu_G    = params(1)
    sigma_G = params(2)
    Lambda(1:M) = params(3:2+M)
    theta(1:M)  = params(3+M:2+2*M)
    Lambda_total = sum(Lambda(1:M))
    expL = exp(-Lambda_total)

    do j = 1, M
      aj(j) = 1.0_dp - theta(1) / theta(j)
    end do

    call compute_bm(aj, Lambda, M, L, bcoeff)
    call compute_dbm(aj, Lambda, theta, M, L, db)
    call compute_Omega(bcoeff, L, Omega)
    call compute_dOmega(bcoeff, db, L, M, Omega, dOmega)

    call accumulate_pdf_jac_vectorized(t, nt, mu_G, sigma_G, theta(1), M, L, Omega, dOmega, expL, f_out, jac_out)
  end subroutine peak_pdf_jac

  subroutine peak_pdf_jac_fitvars(t, nt, xi, nxi, M, L, f_out, jac_out) bind(c, name='peak_pdf_jac_fitvars')
    integer(c_int), intent(in), value :: nt, nxi, M, L
    real(dp), intent(in)  :: t(nt), xi(nxi)
    real(dp), intent(out) :: f_out(nt), jac_out(nxi, nt)

    real(dp) :: params(nxi)
    real(dp) :: sig_val, cumsum
    integer  :: i, j

    call xi_to_physical(xi, nxi, M, params)
    call peak_pdf_jac(t, nt, params, nxi, M, L, f_out, jac_out)

    sig_val = sigmoid(xi(2))
    do i = 1, nt
      jac_out(2, i) = jac_out(2, i) * sig_val
    end do

    do j = 1, M
      sig_val = sigmoid(xi(2 + j))
      do i = 1, nt
        jac_out(2 + j, i) = jac_out(2 + j, i) * sig_val
      end do
    end do

    do i = 1, nt
      cumsum = 0.0_dp
      do j = M, 1, -1
        cumsum = cumsum + jac_out(2 + M + j, i)
        jac_out(2 + M + j, i) = sigmoid(xi(2 + M + j)) * cumsum
      end do
    end do
  end subroutine peak_pdf_jac_fitvars

  subroutine required_L_fitvars(xi, nxi, M, eps_L, L_cap, L_req, deficit) bind(c, name='required_L_fitvars')
    integer(c_int), intent(in), value :: nxi, M, L_cap
    real(dp), intent(in), value :: eps_L
    real(dp), intent(in) :: xi(nxi)
    integer(c_int), intent(out) :: L_req
    real(dp), intent(out) :: deficit
    real(dp) :: params(nxi)

    call xi_to_physical(xi, nxi, M, params)
    call required_L_from_physical(params, nxi, M, eps_L, L_cap, L_req, deficit)
  end subroutine required_L_fitvars

  subroutine required_L_from_physical(params, npar, M, eps_L, L_cap, L_req, deficit)
    integer(c_int), intent(in) :: npar, M, L_cap
    real(dp), intent(in) :: eps_L
    real(dp), intent(in) :: params(npar)
    integer(c_int), intent(out) :: L_req
    real(dp), intent(out) :: deficit

    real(dp) :: Lambda(M), theta(M), aj(M), Lambda_total, expL_neg
    real(dp), allocatable :: bcoeff(:), Omega(:)
    real(dp) :: S_L, s, inv_ell
    integer  :: ell, idx_m, j

    Lambda(1:M) = params(3:2+M)
    theta(1:M)  = params(3+M:2+2*M)
    Lambda_total = sum(Lambda(1:M))
    expL_neg = exp(-Lambda_total)

    do j = 1, M
      aj(j) = 1.0_dp - theta(1) / theta(j)
    end do

    allocate(bcoeff(L_cap), Omega(0:L_cap))
    call compute_bm(aj, Lambda, M, L_cap, bcoeff)

    Omega(0) = 1.0_dp
    S_L = expL_neg
    if (1.0_dp - S_L < eps_L) then
      L_req = 0
      deficit = 1.0_dp - S_L
      deallocate(bcoeff, Omega)
      return
    end if

    do ell = 1, L_cap
      s = 0.0_dp
      do idx_m = 1, ell
        s = s + real(idx_m, dp) * bcoeff(idx_m) * Omega(ell - idx_m)
      end do
      inv_ell = 1.0_dp / real(ell, dp)
      Omega(ell) = s * inv_ell
      S_L = S_L + expL_neg * Omega(ell)
      if (1.0_dp - S_L < eps_L) then
        L_req = ell
        deficit = 1.0_dp - S_L
        deallocate(bcoeff, Omega)
        return
      end if
    end do

    L_req = L_cap
    deficit = 1.0_dp - S_L
    deallocate(bcoeff, Omega)
  end subroutine required_L_from_physical

  subroutine compute_bm(aj, Lambda, M, L, b)
    integer, intent(in) :: M, L
    real(dp), intent(in)  :: aj(M), Lambda(M)
    real(dp), intent(out) :: b(L)
    integer  :: idx_m, j
    real(dp) :: aj_pow

    b = 0.0_dp
    do j = 1, M
      aj_pow = 1.0_dp
      do idx_m = 1, L
        b(idx_m) = b(idx_m) + Lambda(j) * (1.0_dp - aj(j)) * aj_pow
        aj_pow = aj_pow * aj(j)
      end do
    end do
  end subroutine compute_bm

  subroutine compute_dbm(aj, Lambda, theta, M, L, db)
    integer, intent(in) :: M, L
    real(dp), intent(in)  :: aj(M), Lambda(M), theta(M)
    real(dp), intent(out) :: db(L, 2*M)
    integer  :: idx_m, j
    real(dp) :: aj_pow_m1, aj_pow_m2

    db = 0.0_dp
    do j = 1, M
      aj_pow_m2 = 0.0_dp
      aj_pow_m1 = 1.0_dp
      do idx_m = 1, L
        db(idx_m, j) = (1.0_dp - aj(j)) * aj_pow_m1
        if (j >= 2) then
          db(idx_m, M + j) = Lambda(j) * theta(1) / theta(j)**2 * &
               (real(idx_m - 1, dp) * aj_pow_m2 - real(idx_m, dp) * aj_pow_m1)
        end if
        aj_pow_m2 = aj_pow_m1
        aj_pow_m1 = aj_pow_m1 * aj(j)
      end do
    end do

    db(:, M + 1) = 0.0_dp
    do j = 2, M
      aj_pow_m2 = 0.0_dp
      aj_pow_m1 = 1.0_dp
      do idx_m = 1, L
        db(idx_m, M + 1) = db(idx_m, M + 1) + (Lambda(j) / theta(j)) * &
             (real(idx_m, dp) * aj_pow_m1 - real(idx_m - 1, dp) * aj_pow_m2)
        aj_pow_m2 = aj_pow_m1
        aj_pow_m1 = aj_pow_m1 * aj(j)
      end do
    end do
  end subroutine compute_dbm

  subroutine compute_Omega(b, L, Omega)
    integer, intent(in) :: L
    real(dp), intent(in)  :: b(L)
    real(dp), intent(out) :: Omega(0:L)
    integer  :: ell, idx_m
    real(dp) :: s, inv_ell

    Omega(0) = 1.0_dp
    do ell = 1, L
      s = 0.0_dp
      do idx_m = 1, ell
        s = s + real(idx_m, dp) * b(idx_m) * Omega(ell - idx_m)
      end do
      inv_ell = 1.0_dp / real(ell, dp)
      Omega(ell) = s * inv_ell
    end do
  end subroutine compute_Omega

  subroutine compute_dOmega(b, db, L, M, Omega, dOmega)
    integer, intent(in) :: L, M
    real(dp), intent(in)  :: b(L), db(L, 2*M), Omega(0:L)
    real(dp), intent(out) :: dOmega(0:L, 2*M)
    integer  :: ell, idx_m, p
    real(dp) :: s, inv_ell

    dOmega(0, :) = 0.0_dp
    do p = 1, 2 * M
      do ell = 1, L
        s = 0.0_dp
        do idx_m = 1, ell
          s = s + real(idx_m, dp) * (db(idx_m, p) * Omega(ell - idx_m) + b(idx_m) * dOmega(ell - idx_m, p))
        end do
        inv_ell = 1.0_dp / real(ell, dp)
        dOmega(ell, p) = s * inv_ell
      end do
    end do
  end subroutine compute_dOmega

  subroutine accumulate_pdf_vectorized(t, nt, mu_G, sigma_G, theta1, L, Omega, expL, f_out)
    integer, intent(in) :: nt, L
    real(dp), intent(in) :: t(nt), mu_G, sigma_G, theta1, Omega(0:L), expL
    real(dp), intent(out) :: f_out(nt)

    real(dp) :: sig2, rat_sig_th, coeff_common, inv_e
    real(dp), allocatable :: z(:), gx(:), aa(:), cr(:), h_m2(:), h_m1(:), h_cur(:), sum_f(:)
    integer :: i, ell

    sig2 = sigma_G**2
    rat_sig_th = sig2 / theta1**2

    allocate(z(nt), gx(nt), aa(nt), cr(nt), h_m2(nt), h_m1(nt), h_cur(nt), sum_f(nt))

    do i = 1, nt
      z(i) = t(i) - mu_G
      gx(i) = exp(-0.5_dp * z(i) * z(i) / sig2)
      h_m2(i) = gx(i) / (sigma_G * SQRT2PI)
      sum_f(i) = Omega(0) * h_m2(i)
    end do

    if (L >= 1) then
      coeff_common = 0.5_dp / theta1
      do i = 1, nt
        aa(i) = (mu_G + sig2 / theta1 - t(i)) / (sigma_G * SQRT2)
        if (aa(i) > -ERFCX_THR) then
          h_m1(i) = coeff_common * gx(i) * erfc_scaled(aa(i))
        else
          h_m1(i) = coeff_common * exp(-z(i) / theta1 + 0.5_dp * rat_sig_th) * erfc(aa(i))
        end if
        cr(i) = (z(i) - sig2 / theta1) / theta1
        sum_f(i) = sum_f(i) + Omega(1) * h_m1(i)
      end do
    end if

    do ell = 2, L
      inv_e = 1.0_dp / real(ell - 1, dp)
      do i = 1, nt
        h_cur(i) = (rat_sig_th * h_m2(i) + cr(i) * h_m1(i)) * inv_e
        sum_f(i) = sum_f(i) + Omega(ell) * h_cur(i)
      end do
      h_m2 = h_m1
      h_m1 = h_cur
    end do

    do i = 1, nt
      f_out(i) = expL * sum_f(i)
    end do

    deallocate(z, gx, aa, cr, h_m2, h_m1, h_cur, sum_f)
  end subroutine accumulate_pdf_vectorized

  subroutine accumulate_pdf_jac_vectorized(t, nt, mu_G, sigma_G, theta1, M, L, Omega, dOmega, expL, f_out, jac_out)
    integer, intent(in) :: nt, M, L
    real(dp), intent(in) :: t(nt), mu_G, sigma_G, theta1, Omega(0:L), dOmega(0:L,2*M), expL
    real(dp), intent(out) :: f_out(nt), jac_out(2+2*M, nt)

    real(dp) :: sig2, rat_sig_th, pref, inv_e
    real(dp) :: da_dmu, da_dth1, drat_dsig, drat_dth1, dcoeff_dmu, dcoeff_dsig
    real(dp) :: const_dcoeff_dth1
    real(dp), allocatable :: z(:), gx(:), aa(:), cr(:), dcoeff_dth1(:)
    real(dp), allocatable :: h_m2(:), h_m1(:), h_cur(:)
    real(dp), allocatable :: dhmu_m2(:), dhmu_m1(:), dhmu_cur(:)
    real(dp), allocatable :: dhsig_m2(:), dhsig_m1(:), dhsig_cur(:)
    real(dp), allocatable :: dhth_m2(:), dhth_m1(:), dhth_cur(:)
    real(dp), allocatable :: sum_f(:), acc_mu(:), acc_sig(:), acc_th1h(:), acc_domega(:,:)
    real(dp) :: E, dE_da, Q, dQ_mu, dQ_sig, dQ_th1, derfc_da, da_dsig_i
    integer :: i, ell, j, col, p

    sig2 = sigma_G**2
    rat_sig_th = sig2 / theta1**2
    pref = 0.5_dp / theta1
    da_dmu = 1.0_dp / (sigma_G * SQRT2)
    da_dth1 = -sigma_G / (theta1**2 * SQRT2)
    drat_dsig = 2.0_dp * sigma_G / theta1**2
    drat_dth1 = -2.0_dp * sig2 / theta1**3
    dcoeff_dmu = -1.0_dp / theta1
    dcoeff_dsig = -2.0_dp * sigma_G / theta1**2
    const_dcoeff_dth1 = 2.0_dp * sig2 / theta1**3

    allocate(z(nt), gx(nt), aa(nt), cr(nt), dcoeff_dth1(nt))
    allocate(h_m2(nt), h_m1(nt), h_cur(nt))
    allocate(dhmu_m2(nt), dhmu_m1(nt), dhmu_cur(nt))
    allocate(dhsig_m2(nt), dhsig_m1(nt), dhsig_cur(nt))
    allocate(dhth_m2(nt), dhth_m1(nt), dhth_cur(nt))
    allocate(sum_f(nt), acc_mu(nt), acc_sig(nt), acc_th1h(nt), acc_domega(nt,2*M))

    acc_domega = 0.0_dp
    jac_out = 0.0_dp

    do i = 1, nt
      z(i) = t(i) - mu_G
      gx(i) = exp(-0.5_dp * z(i) * z(i) / sig2)
      h_m2(i) = gx(i) / (sigma_G * SQRT2PI)
      dhmu_m2(i) = (z(i) / sig2) * h_m2(i)
      dhsig_m2(i) = ((z(i) * z(i) / (sigma_G * sig2)) - 1.0_dp / sigma_G) * h_m2(i)
      dhth_m2(i) = 0.0_dp

      sum_f(i) = Omega(0) * h_m2(i)
      acc_mu(i) = Omega(0) * dhmu_m2(i)
      acc_sig(i) = Omega(0) * dhsig_m2(i)
      acc_th1h(i) = 0.0_dp
    end do

    if (L >= 1) then
      do i = 1, nt
        aa(i) = (mu_G + sig2 / theta1 - t(i)) / (sigma_G * SQRT2)
        da_dsig_i = (1.0_dp / theta1 + z(i) / sig2) / SQRT2

        if (aa(i) > -ERFCX_THR) then
          E = erfc_scaled(aa(i))
          dE_da = 2.0_dp * aa(i) * E - 2.0_dp / SQRTPI
          h_m1(i) = pref * gx(i) * E
          dhmu_m1(i) = pref * (gx(i) * (z(i) / sig2) * E + gx(i) * dE_da * da_dmu)
          dhsig_m1(i) = pref * (gx(i) * (z(i) * z(i) / (sigma_G * sig2)) * E + gx(i) * dE_da * da_dsig_i)
          dhth_m1(i) = -h_m1(i) / theta1 + pref * gx(i) * dE_da * da_dth1
        else
          Q = exp(-z(i) / theta1 + 0.5_dp * rat_sig_th)
          derfc_da = -2.0_dp / SQRTPI * exp(-aa(i) * aa(i))
          h_m1(i) = pref * Q * erfc(aa(i))
          dQ_mu = Q / theta1
          dQ_sig = Q * sigma_G / theta1**2
          dQ_th1 = Q * (z(i) / theta1**2 - sig2 / theta1**3)
          dhmu_m1(i) = pref * (dQ_mu * erfc(aa(i)) + Q * derfc_da * da_dmu)
          dhsig_m1(i) = pref * (dQ_sig * erfc(aa(i)) + Q * derfc_da * da_dsig_i)
          dhth_m1(i) = -h_m1(i) / theta1 + pref * (dQ_th1 * erfc(aa(i)) + Q * derfc_da * da_dth1)
        end if

        cr(i) = (z(i) - sig2 / theta1) / theta1
        dcoeff_dth1(i) = -z(i) / theta1**2 + const_dcoeff_dth1

        sum_f(i) = sum_f(i) + Omega(1) * h_m1(i)
        acc_mu(i) = acc_mu(i) + Omega(1) * dhmu_m1(i)
        acc_sig(i) = acc_sig(i) + Omega(1) * dhsig_m1(i)
        acc_th1h(i) = acc_th1h(i) + Omega(1) * dhth_m1(i)
      end do

      do p = 1, 2*M
        if (dOmega(1,p) /= 0.0_dp) then
          do i = 1, nt
            acc_domega(i,p) = acc_domega(i,p) + dOmega(1,p) * h_m1(i)
          end do
        end if
      end do
    end if

    do ell = 2, L
      inv_e = 1.0_dp / real(ell - 1, dp)
      do i = 1, nt
        h_cur(i) = (rat_sig_th * h_m2(i) + cr(i) * h_m1(i)) * inv_e
        dhmu_cur(i) = (rat_sig_th * dhmu_m2(i) + dcoeff_dmu * h_m1(i) + cr(i) * dhmu_m1(i)) * inv_e
        dhsig_cur(i) = ((drat_dsig * h_m2(i) + rat_sig_th * dhsig_m2(i)) + &
                        (dcoeff_dsig * h_m1(i) + cr(i) * dhsig_m1(i))) * inv_e
        dhth_cur(i) = ((drat_dth1 * h_m2(i) + rat_sig_th * dhth_m2(i)) + &
                       (dcoeff_dth1(i) * h_m1(i) + cr(i) * dhth_m1(i))) * inv_e

        sum_f(i) = sum_f(i) + Omega(ell) * h_cur(i)
        acc_mu(i) = acc_mu(i) + Omega(ell) * dhmu_cur(i)
        acc_sig(i) = acc_sig(i) + Omega(ell) * dhsig_cur(i)
        acc_th1h(i) = acc_th1h(i) + Omega(ell) * dhth_cur(i)
      end do

      do p = 1, 2*M
        if (dOmega(ell,p) /= 0.0_dp) then
          do i = 1, nt
            acc_domega(i,p) = acc_domega(i,p) + dOmega(ell,p) * h_cur(i)
          end do
        end if
      end do

      h_m2 = h_m1
      h_m1 = h_cur
      dhmu_m2 = dhmu_m1
      dhmu_m1 = dhmu_cur
      dhsig_m2 = dhsig_m1
      dhsig_m1 = dhsig_cur
      dhth_m2 = dhth_m1
      dhth_m1 = dhth_cur
    end do

    do i = 1, nt
      f_out(i) = expL * sum_f(i)
      jac_out(1, i) = expL * acc_mu(i)
      jac_out(2, i) = expL * acc_sig(i)
    end do

    do j = 1, M
      col = 2 + j
      do i = 1, nt
        jac_out(col, i) = expL * (acc_domega(i, j) - sum_f(i))
      end do
    end do

    col = 2 + M + 1
    do i = 1, nt
      jac_out(col, i) = expL * (acc_domega(i, M + 1) + acc_th1h(i))
    end do

    do j = 2, M
      col = 2 + M + j
      do i = 1, nt
        jac_out(col, i) = expL * acc_domega(i, M + j)
      end do
    end do

    deallocate(z, gx, aa, cr, dcoeff_dth1)
    deallocate(h_m2, h_m1, h_cur)
    deallocate(dhmu_m2, dhmu_m1, dhmu_cur)
    deallocate(dhsig_m2, dhsig_m1, dhsig_cur)
    deallocate(dhth_m2, dhth_m1, dhth_cur)
    deallocate(sum_f, acc_mu, acc_sig, acc_th1h, acc_domega)
  end subroutine accumulate_pdf_jac_vectorized

end module chromatopeak_core
